"""Async wrappers around Groq (STT + streaming LLM) and ElevenLabs / Sarvam (TTS).

All calls share one httpx.AsyncClient (keep-alive, HTTP/2 when available) so there is no
TCP/TLS handshake on the critical path after the first request. `warm_up()` opens those
connections at server start.
"""
import asyncio
import base64
import json
import re
from typing import AsyncIterator

import httpx

from . import config

GROQ_BASE = "https://api.groq.com/openai/v1"
ELEVEN_BASE = "https://api.elevenlabs.io/v1"
SARVAM_URL = "https://api.sarvam.ai/text-to-speech"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# ElevenLabs free/starter plans allow only 2 parallel requests; more gives a 429 error.
_ELEVEN_SLOTS = asyncio.Semaphore(2)


class ProviderError(RuntimeError):
    """Raised when an upstream API fails; message is safe to show the user."""


async def hedged(make, after: float):
    """Run make(); if it has not finished after `after` seconds, start a second identical call and
    use whichever succeeds first. Cuts the slow tail (a stalled connection or a busy server)."""
    tasks = [asyncio.ensure_future(make())]
    try:
        done, _ = await asyncio.wait(tasks, timeout=after)
        if not done or tasks[0].exception() is not None:  # slow, or failed outright: try again now
            tasks.append(asyncio.ensure_future(make()))
        pending, last = set(tasks), None
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                if d.cancelled():
                    continue
                if d.exception() is None:
                    return d.result()
                last = d.exception()
        raise last or ProviderError("request failed")
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


def make_client() -> httpx.AsyncClient:
    try:
        import h2  # noqa: F401
        http2 = True
    except ImportError:
        http2 = False
    return httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=3.0),
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=300),
        http2=http2,
    )


async def warm_up(client: httpx.AsyncClient) -> None:
    """Open connections (DNS + TLS) ahead of the first user turn. Errors are ignored."""
    hosts = []
    if config.GROQ_API_KEY:
        hosts.append(f"{GROQ_BASE}/models")
    if config.ELEVENLABS_API_KEY:
        hosts.append(f"{ELEVEN_BASE}/models")
    if config.SARVAM_API_KEY:
        hosts.append("https://api.sarvam.ai/")
    for url in hosts:
        try:
            await client.get(url, headers={"Authorization": f"Bearer {config.GROQ_API_KEY}",
                                           "xi-api-key": config.ELEVENLABS_API_KEY})
        except Exception:
            pass


def _raise_for(resp: httpx.Response, name: str) -> None:
    if resp.status_code >= 400:
        # Never echo request headers (they contain keys); body is the provider's error text.
        detail = resp.text[:300].replace("\n", " ")
        raise ProviderError(f"{name} returned {resp.status_code}: {detail}")


async def transcribe(client: httpx.AsyncClient, audio: bytes, filename: str, content_type: str) -> str:
    if not config.GROQ_API_KEY:
        raise ProviderError("GROQ_API_KEY is not set")
    data = {"model": config.GROQ_STT_MODEL, "response_format": "json", "temperature": "0"}
    if config.STT_LANGUAGE:
        data["language"] = config.STT_LANGUAGE  # skips language detection (a little faster)
    resp = await client.post(
        f"{GROQ_BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        files={"file": (filename, audio, content_type)},
        data=data,
    )
    _raise_for(resp, "Groq speech-to-text")
    return (resp.json().get("text") or "").strip()


async def chat_stream(client: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[str]:
    """Yield the LLM answer token by token (Groq, OpenAI-compatible streaming)."""
    if not config.GROQ_API_KEY:
        raise ProviderError("GROQ_API_KEY is not set")
    async with client.stream(
        "POST",
        f"{GROQ_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        json={
            "model": config.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": config.LLM_MAX_TOKENS,
            "stream": True,
        },
    ) as resp:
        if resp.status_code >= 400:
            body = (await resp.aread()).decode(errors="replace")[:300].replace("\n", " ")
            raise ProviderError(f"Groq LLM returned {resp.status_code}: {body}")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if delta:
                yield delta


async def synthesize(client: httpx.AsyncClient, text: str, provider: str) -> tuple[bytes, str]:
    """Return (audio_bytes, mime_type) for one short piece of text."""
    if provider == "elevenlabs":
        if not config.ELEVENLABS_API_KEY:
            raise ProviderError("ELEVENLABS_API_KEY is not set")
        resp = None
        for attempt in range(3):  # retry a busy (429) or dropped connection instead of failing
            try:
                async with _ELEVEN_SLOTS:
                    resp = await client.post(
                        f"{ELEVEN_BASE}/text-to-speech/{config.ELEVENLABS_VOICE_ID}",
                        params={"output_format": config.ELEVENLABS_FORMAT, "optimize_streaming_latency": 3},
                        headers={"xi-api-key": config.ELEVENLABS_API_KEY, "Accept": "audio/mpeg"},
                        json={"text": text, "model_id": config.ELEVENLABS_TTS_MODEL},
                    )
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError):
                if attempt == 2:
                    raise
                await asyncio.sleep(0.05)
                continue
            if resp.status_code == 429 and attempt < 2:
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            break
        _raise_for(resp, "ElevenLabs")
        return resp.content, "audio/mpeg"

    if provider == "sarvam":
        if not config.SARVAM_API_KEY:
            raise ProviderError("SARVAM_API_KEY is not set")
        language = "hi-IN" if _DEVANAGARI.search(text) else "en-IN"
        resp = await client.post(
            SARVAM_URL,
            headers={"api-subscription-key": config.SARVAM_API_KEY},
            json={
                "text": text[:1500],
                "target_language_code": language,
                "speaker": config.SARVAM_SPEAKER,
                "model": config.SARVAM_MODEL,
                "speech_sample_rate": config.SARVAM_SAMPLE_RATE,
            },
        )
        _raise_for(resp, "Sarvam AI")
        audios = resp.json().get("audios") or []
        if not audios:
            raise ProviderError("Sarvam AI returned no audio")
        return base64.b64decode(audios[0]), "audio/wav"

    raise ProviderError(f"Unknown TTS provider '{provider}'")
