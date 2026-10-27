"""FastAPI voice bot for a cafe seat-booking desk: Groq (speech-to-text + streaming LLM) and ElevenLabs / Sarvam (text-to-speech).

Speed design: the LLM streams its answer, every finished sentence is sent to text-to-speech
immediately (in parallel), and audio chunks are streamed to the browser as they become ready.
The first sound therefore plays after STT + first LLM sentence + first TTS chunk, not after the
whole answer. Every turn is saved with metadata and written to the answer-trace log.
"""
import asyncio
import base64
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from datetime import date

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import cafe, chatlog, config, metadata, providers

logging.basicConfig(
    filename=config.LOG_DIR / "voicebot.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("voicebot")

# Answer-trace log: one JSON line per turn saying WHICH FUNCTION produced the answer and which
# functions ran (with timings). Kept separate from voicebot.log so it is easy to read or ship.
trace_log = logging.getLogger("voicebot.trace")
trace_log.setLevel(logging.INFO)
trace_log.propagate = False
if not trace_log.handlers:
    _h = RotatingFileHandler(config.TRACE_LOG, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(message)s"))
    trace_log.addHandler(_h)

STATIC_DIR = Path(__file__).parent / "static"
EXIT_PHRASES = ("stop voice bot", "exit voice bot", "बंद करो", "बंद कर दो", "अलविदा")

# Function names written to the trace log as "answer_function".
FN_LLM = "providers.chat_stream"
FN_EXIT = "main.exit_handler"
FN_EMPTY = "main.empty_transcription_fallback"
FN_LLM_EMPTY = "main.llm_empty_reply_fallback"
# Booking functions (cafe.py) report their own name, e.g. cafe.booking_flow, cafe.availability, cafe.hours.

# After the first (short) chunk, sentences are merged until this many characters, so a normal
# answer costs 2 voice requests instead of 3-4 (less total voice time, fewer rate-limit hits).
REST_CHUNK_CHARS = 110

sessions: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    metadata.init_db()
    cafe.init_db()
    app.state.http = providers.make_client()

    async def keep_warm():
        # Re-ping the providers so connections never go cold (a cold TLS handshake costs 100-300 ms).
        while True:
            await providers.warm_up(app.state.http)
            await asyncio.sleep(25)

    warm = asyncio.create_task(keep_warm())
    yield
    warm.cancel()
    await app.state.http.aclose()


app = FastAPI(title="Voice Bot", lifespan=lifespan)


# ---------------------------------------------------------------- helpers
def _history(session_id: str) -> list[dict]:
    if session_id not in sessions:
        # After a server restart (or when reopening an old conversation) rebuild context from the database.
        sessions[session_id] = [{"role": "system", "content": cafe.system_prompt()}] + metadata.llm_context(
            session_id, config.MAX_HISTORY_MESSAGES
        )
    return sessions[session_id]


def _trim(history: list[dict]) -> None:
    if len(history) > config.MAX_HISTORY_MESSAGES + 1:
        history[:] = [history[0]] + history[-config.MAX_HISTORY_MESSAGES:]


def _pick_tts(requested: str | None) -> str:
    available = config.available_tts_providers()
    if requested and requested in available:
        return requested
    if config.TTS_PROVIDER in available:
        return config.TTS_PROVIDER
    return available[0]


def _r(value: float) -> float:
    return round(value, 3)


_STRIP = re.compile(r"[*#`_~>]+")


def _clean(text: str) -> str:
    """Remove markdown characters the voice would otherwise try to read out."""
    return _STRIP.sub("", text).strip()


def split_sentences(buf: str, first: bool) -> tuple[list[str], str]:
    """Cut finished sentences off the front of a growing text buffer.

    A sentence ends at . ! ? or the Hindi danda followed by a space. For the very first chunk we
    also allow a break at a comma after ~8 characters (or a word boundary after ~13) so the voice can start even sooner.
    """
    out: list[str] = []
    start = 0
    for i, ch in enumerate(buf):
        nxt = buf[i + 1] if i + 1 < len(buf) else ""
        end = ch in ".!?।" and nxt.isspace()
        # first chunk: break at a comma after ~8 chars, or at any word boundary after ~13 chars
        early = first and not out and nxt.isspace() and (
            (ch in ",;:" and (i - start) >= 8) or (ch.isalnum() and (i - start) >= 13)
        )
        if end or early:
            piece = buf[start:i + 1].strip()
            if piece:
                out.append(piece)
            start = i + 1
    return out, buf[start:]


async def _turn_events(
    http,
    *,
    entry: str,
    session_id: str,
    input_mode: str,
    audio: tuple[bytes, str, str] | None,
    text: str | None,
    tts_provider: str | None,
    request_start: float,
):
    """Run STT -> streaming LLM -> parallel TTS for one turn and yield events for the browser.

    Events: transcript, text (one sentence), audio (one chunk), done | error.
    request_start is a time.perf_counter() value taken when the request arrived.
    """
    provider = _pick_tts(tts_provider)
    record = {
        "turn_id": str(uuid.uuid4()),
        "session_id": session_id,
        **metadata.now_parts(),  # date/time the user's question reached the server
        "input_mode": input_mode,
        "user_input": "",
        "bot_response": "",
        "stt_seconds": 0.0,
        "llm_seconds": 0.0,
        "tts_seconds": 0.0,
        "llm_first_sentence_seconds": 0.0,
        "tts_first_seconds": 0.0,
        "server_first_audio_seconds": None,
        "server_total_seconds": 0.0,
        "answer_function": "",
        "tts_provider": provider,
        "llm_model": config.GROQ_MODEL,
        "status": "started",
        "error": "",
    }
    chain: list[dict] = [{"fn": entry, "seconds": None}, {"fn": "main._turn_events", "seconds": None}]
    exiting = False
    history_user_added = False
    completed = False
    tasks: list[asyncio.Task] = []
    producer: asyncio.Task | None = None
    parts: list[str] = []

    try:
        # 1. Speech to text (skipped for typed input)
        if audio is not None:
            t0 = time.perf_counter()
            data, filename, content_type = audio
            text = await providers.hedged(
                lambda: providers.transcribe(http, data, filename, content_type), config.STT_HEDGE_SECONDS)
            record["stt_seconds"] = _r(time.perf_counter() - t0)
            chain.append({"fn": "providers.transcribe", "seconds": record["stt_seconds"]})
        record["user_input"] = text or ""
        yield {"type": "transcript", "text": record["user_input"], "turn_id": record["turn_id"],
               "date": record["date"], "time": record["time"], "stt_seconds": record["stt_seconds"],
               "input_mode": input_mode}

        # 2. Pick the function that produces the answer
        async def fixed(sentence: str):
            parts.append(sentence)
            yield sentence

        suffix: str | None = None   # booking question spoken after an LLM answer given in the middle of a booking

        async def cafe_sentences(reply_text: str):
            """A booking-desk reply is fixed text: voice the first sentence alone, then the rest in one chunk."""
            history = _history(session_id)
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": reply_text})
            _trim(history)
            pieces = [p for p in re.split(r"(?<=[.!?।])\s+", reply_text.strip()) if p]
            first_piece, rest = (pieces[0], " ".join(pieces[1:])) if pieces else (reply_text, "")
            for piece in (first_piece, rest):
                if piece:
                    parts.append(piece)
                    yield piece

        async def llm_sentences():
            nonlocal history_user_added
            history = _history(session_id)
            history.append({"role": "user", "content": text})
            history_user_added = True
            t0 = time.perf_counter()
            buf, first = "", True
            pend = ""  # after the short first chunk, the rest is voiced in as few big chunks as possible

            async for token in providers.chat_stream(http, history):
                buf += token
                sentences, buf = split_sentences(buf, first)
                for sent in sentences:
                    sent = _clean(sent)
                    if not sent:
                        continue
                    if first:
                        record["llm_first_sentence_seconds"] = _r(time.perf_counter() - t0)
                        first = False
                        parts.append(sent)
                        yield sent  # chunk 0: voiced immediately for the fastest first sound
                        continue
                    pend = (pend + " " + sent).strip()
                    if len(pend) >= REST_CHUNK_CHARS:
                        parts.append(pend)
                        yield pend
                        pend = ""
            tail = _clean(buf)
            if tail:
                if first:
                    record["llm_first_sentence_seconds"] = _r(time.perf_counter() - t0)
                    first = False
                pend = (pend + " " + tail).strip()
            if pend:
                parts.append(pend)
                yield pend
            if suffix:  # booking was in progress: after answering, ask the pending booking question again
                parts.append(suffix)
                yield suffix
            record["llm_seconds"] = _r(time.perf_counter() - t0)
            chain.append({"fn": FN_LLM, "seconds": record["llm_seconds"]})
            if not parts:
                record["answer_function"] = FN_LLM_EMPTY
                parts.append("Sorry, I could not generate an answer.")
                yield parts[0]
            history.append({"role": "assistant", "content": " ".join(parts)})
            history_user_added = False
            _trim(history)

        if not text:
            record["answer_function"], record["status"] = FN_EMPTY, "empty_transcription"
            source = fixed("Sorry, I could not understand. Please try again.")
        elif any(p in text.lower() for p in EXIT_PHRASES):
            record["answer_function"], record["status"] = FN_EXIT, "exit"
            exiting = True
            cafe.reset(session_id)
            source = fixed("Goodbye! फिर मिलेंगे।")
        else:
            booking = cafe.handle(session_id, text, via_voice=(input_mode == "voice"))
            if booking is not None and booking.text is not None:   # answered from the seat database
                record["answer_function"], record["status"] = booking.fn, "success"
                source = cafe_sentences(booking.text)
            else:                                                  # everything else: the LLM, as before
                suffix = booking.suffix if booking is not None else None
                record["answer_function"], record["status"] = FN_LLM, "success"
                source = llm_sentences()
        if record["answer_function"] != FN_LLM:  # the LLM path adds its own timed entry
            chain.append({"fn": record["answer_function"], "seconds": None})

        # 3. Pipeline: producer feeds sentences to TTS tasks (run in parallel); we yield in order.
        queue: asyncio.Queue = asyncio.Queue()

        use_browser = provider == "browser"  # one voice per reply: flips to True only if the FIRST chunk fails

        async def synth(sentence: str, idx: int):
            t0 = time.perf_counter()
            if use_browser:  # spoken by the browser itself, nothing to generate here
                return b"", "", 0.0
            if idx == 0:  # the chunk the user waits for: hedge it so one slow call cannot hold up the reply
                audio_bytes, mime = await providers.hedged(
                    lambda: providers.synthesize(http, sentence, provider), config.TTS_HEDGE_SECONDS)
            else:
                audio_bytes, mime = await providers.synthesize(http, sentence, provider)
            return audio_bytes, mime, _r(time.perf_counter() - t0)

        async def produce():
            idx = 0
            try:
                async for sentence in source:
                    task = asyncio.create_task(synth(sentence, idx))
                    tasks.append(task)
                    queue.put_nowait((sentence, idx, task))
                    idx += 1
            except Exception as exc:  # LLM failure: surface it to the consumer
                queue.put_nowait(exc)
            finally:
                queue.put_nowait(None)

        producer = asyncio.create_task(produce())
        tts_failed = False
        played_server_audio = False
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            sentence, idx, task = item
            yield {"type": "text", "text": sentence, "index": idx}
            if use_browser:  # whole reply is spoken by the browser voice
                if record["server_first_audio_seconds"] is None:
                    record["server_first_audio_seconds"] = _r(time.perf_counter() - request_start)
                yield {"type": "speak", "text": sentence, "index": idx}
                continue
            try:
                audio_bytes, mime, secs = await task
            except Exception as exc:
                if not tts_failed:
                    record["status"] = "tts_error"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    log.error("TTS failed: %s", record["error"])
                tts_failed = True
                if not played_server_audio:
                    use_browser = True  # nothing played yet: switch the entire reply to browser voice
                    yield {"type": "speak", "text": sentence, "index": idx}
                continue  # later chunk failed: skip it rather than mix in a second voice
            played_server_audio = True
            record["tts_seconds"] = _r(record["tts_seconds"] + secs)  # total voice work (chunks overlap)
            chain.append({"fn": f"providers.synthesize[{provider}]#{idx}", "seconds": secs})
            if idx == 0:
                record["tts_first_seconds"] = secs
            if record["server_first_audio_seconds"] is None:
                record["server_first_audio_seconds"] = _r(time.perf_counter() - request_start)
            yield {"type": "audio", "index": idx, "mime": mime,
                   "data": base64.b64encode(audio_bytes).decode()}
        record["bot_response"] = " ".join(parts)
        completed = True

    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["bot_response"] = " ".join(parts)
        log.exception("Turn failed")
    finally:
        if producer is not None and not producer.done():
            producer.cancel()
        for t in tasks:
            if not t.done():
                t.cancel()
        if history_user_added:  # LLM failed or the browser left mid-answer: drop the dangling question
            h = sessions.get(session_id)
            if h and h[-1].get("role") == "user":
                h.pop()

    if not completed and record["status"] != "error":
        return  # browser disconnected mid-turn: nothing more to send or save

    record["server_total_seconds"] = _r(time.perf_counter() - request_start)
    stages = {"speech_to_text": record["stt_seconds"], "llm_first_sentence": record["llm_first_sentence_seconds"],
              "voice_first_chunk": record["tts_first_seconds"]}
    first_audio = record["server_first_audio_seconds"] or 0
    # name the stage that took longest whenever the first sound was later than ~1.2 s on the server
    record["slowest_stage"] = max(stages, key=stages.get) if first_audio > 1.2 else ""
    record["call_chain"] = " > ".join(
        c["fn"] + (f"({c['seconds']}s)" if c["seconds"] is not None else "") for c in chain
    )
    metadata.touch_conversation(session_id, record["user_input"], record["timestamp_iso"])
    metadata.save_turn(record)
    chatlog.write(session_id)
    trace_log.info(json.dumps({
        "timestamp": record["timestamp_iso"],
        "turn_id": record["turn_id"],
        "session_id": session_id,
        "user_input": record["user_input"],
        "answer_function": record["answer_function"] or "(none: failed before an answer)",
        "bot_response": record["bot_response"],
        "call_chain": chain,
        "llm_model": record["llm_model"],
        "tts_provider": provider,
        "seconds": {k: record[k] for k in (
            "stt_seconds", "llm_first_sentence_seconds", "llm_seconds", "tts_first_seconds",
            "tts_seconds", "server_first_audio_seconds", "server_total_seconds")},
        "slowest_stage": record["slowest_stage"],
        "status": record["status"],
        "error": record["error"],
    }, ensure_ascii=False))
    log.info(
        "turn=%s fn=%s status=%s stt=%.3f llm_first=%.3f tts_first=%.3f first_audio=%s total=%.3f",
        record["turn_id"], record["answer_function"], record["status"], record["stt_seconds"],
        record["llm_first_sentence_seconds"], record["tts_first_seconds"],
        record["server_first_audio_seconds"], record["server_total_seconds"],
    )

    if record["status"] == "error":
        yield {"type": "error", "message": record["error"], "metadata": metadata.get_turn(record["turn_id"])}
    else:
        # expect_reply: a booking is in progress, so the page starts listening again by itself
        yield {"type": "done", "metadata": metadata.get_turn(record["turn_id"]), "exit": exiting,
               "expect_reply": (not exiting) and cafe.is_active(session_id)}


def _ndjson(events) -> StreamingResponse:
    async def body():
        async for ev in events:
            yield (json.dumps(ev, ensure_ascii=False) + "\n").encode()

    return StreamingResponse(
        body(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- API
class TextTurn(BaseModel):
    text: str
    session_id: str = "default"
    tts_provider: str | None = None


class ClientMetrics(BaseModel):
    time_to_audio_seconds: float | None = None
    playback_seconds: float | None = None
    tts_first_seconds: float | None = None  # browser-voice only: speak request -> voice starts
    tts_seconds: float | None = None        # browser-voice only: sum of that delay over all chunks


@app.get("/api/config")
def get_config():
    return {
        "tts_providers": config.available_tts_providers(),
        "default_tts_provider": _pick_tts(None),
        "llm_model": config.GROQ_MODEL,
        "stt_model": config.GROQ_STT_MODEL,
        "groq_ready": bool(config.GROQ_API_KEY),
        "cafe_name": config.CAFE_NAME,
        "voice_only_booking": config.VOICE_ONLY_BOOKING,
    }


@app.post("/api/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    session_id: str = Form("default"),
    tts_provider: str | None = Form(None),
):
    start = time.perf_counter()  # clock starts as soon as the handler runs
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    return _ndjson(_turn_events(
        app.state.http,
        entry="main.chat_voice",
        session_id=session_id,
        input_mode="voice",
        audio=(data, audio.filename or "speech.webm", audio.content_type or "audio/webm"),
        text=None,
        tts_provider=tts_provider,
        request_start=start,
    ))


@app.post("/api/chat/text")
async def chat_text(body: TextTurn):
    start = time.perf_counter()
    return _ndjson(_turn_events(
        app.state.http,
        entry="main.chat_text",
        session_id=body.session_id,
        input_mode="text",
        audio=None,
        text=body.text.strip(),
        tts_provider=body.tts_provider,
        request_start=start,
    ))


@app.get("/api/trace")
def get_trace(limit: int = 50):
    """Most recent answer-trace entries (which function answered each turn), newest first."""
    limit = max(1, min(limit, 500))
    if not config.TRACE_LOG.exists():
        return []
    lines = config.TRACE_LOG.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(l) for l in reversed(lines) if l.strip()]


@app.get("/api/trace.log")
def download_trace():
    text = config.TRACE_LOG.read_text(encoding="utf-8") if config.TRACE_LOG.exists() else ""
    return PlainTextResponse(
        text, media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=answer_trace.jsonl"},
    )


@app.post("/api/turns/{turn_id}/client-metrics")
def post_client_metrics(turn_id: str, body: ClientMetrics):
    """The browser reports what it measured: stop-speaking -> audio start, and playback length."""
    if not metadata.update_client_metrics(
        turn_id,
        None if body.time_to_audio_seconds is None else _r(body.time_to_audio_seconds),
        None if body.playback_seconds is None else _r(body.playback_seconds),
        None if body.tts_first_seconds is None else _r(body.tts_first_seconds),
        None if body.tts_seconds is None else _r(body.tts_seconds),
    ):
        raise HTTPException(status_code=404, detail="Unknown turn")
    turn = metadata.get_turn(turn_id)
    chatlog.write(turn["session_id"])  # refresh the chat log with the browser-measured timings
    return turn


@app.get("/api/turns")
def get_turns(limit: int = 50, session_id: str | None = None):
    return metadata.list_turns(max(1, min(limit, 500)), session_id)


@app.get("/api/turns.csv")
def get_turns_csv(session_id: str | None = None):
    name = f"conversation_{session_id[:8]}.csv" if session_id else "conversation_metadata.csv"
    return PlainTextResponse(
        metadata.to_csv(session_id),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={name}"},
    )


# ---- history management
class Rename(BaseModel):
    title: str


@app.get("/api/conversations")
def get_conversations(q: str | None = None):
    return metadata.list_conversations(q)


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str):
    conv = metadata.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.patch("/api/conversations/{conv_id}")
def rename_conversation(conv_id: str, body: Rename):
    if not metadata.rename_conversation(conv_id, body.title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    chatlog.write(conv_id)
    return metadata.get_conversation(conv_id)


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    sessions.pop(conv_id, None)
    if not metadata.delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    chatlog.delete(conv_id)
    return {"ok": True}


@app.delete("/api/conversations")
def delete_all_conversations():
    sessions.clear()
    deleted = metadata.delete_all()
    chatlog.delete_all()
    return {"ok": True, "deleted": deleted}


@app.get("/api/stats")
def get_stats():
    return metadata.stats()


# ---- cafe: seats, opening hours, availability, bookings
def _admin(token: str | None) -> None:
    if config.ADMIN_TOKEN and token != config.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token required (X-Admin-Token header)")


def _day(value: str | None):
    if not value:
        return cafe.now().date()
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must look like 2026-10-25")


class SeatIn(BaseModel):
    id: str
    name: str
    kind: str
    capacity: int


class SeatPatch(BaseModel):
    name: str | None = None
    kind: str | None = None
    capacity: int | None = None
    active: int | None = None


class HoursIn(BaseModel):
    open: str = "09:00"
    close: str = "22:00"
    closed: bool = False


@app.get("/api/cafe/overview")
def cafe_overview(date: str | None = None):
    """Seats with their bookings for one day, plus opening hours (feeds the page's seat panel)."""
    return cafe.overview(_day(date))


@app.get("/api/cafe/seats")
def cafe_seats():
    return cafe.list_seats()


@app.post("/api/cafe/seats")
def cafe_add_seat(seat: SeatIn, x_admin_token: str | None = Header(None)):
    _admin(x_admin_token)
    if seat.kind not in cafe.KINDS or seat.capacity < 1:
        raise HTTPException(status_code=400, detail="kind must be normal or special, capacity at least 1")
    try:
        cafe.add_seat(seat.id, seat.name, seat.kind, seat.capacity)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not add seat: {exc}")
    return {"ok": True}


@app.patch("/api/cafe/seats/{seat_id}")
def cafe_update_seat(seat_id: str, patch: SeatPatch, x_admin_token: str | None = Header(None)):
    _admin(x_admin_token)
    if not cafe.update_seat(seat_id, **patch.model_dump()):
        raise HTTPException(status_code=404, detail="Seat not found or nothing to change")
    return {"ok": True}


@app.get("/api/cafe/hours")
def cafe_hours():
    return cafe.get_hours()


@app.put("/api/cafe/hours/{weekday}")
def cafe_set_hours(weekday: int, body: HoursIn, x_admin_token: str | None = Header(None)):
    """weekday: 0 = Monday ... 6 = Sunday."""
    _admin(x_admin_token)
    if not 0 <= weekday <= 6:
        raise HTTPException(status_code=400, detail="weekday must be 0 (Monday) to 6 (Sunday)")
    try:
        cafe.set_hours(weekday, body.open, body.close, body.closed)
    except Exception:
        raise HTTPException(status_code=400, detail="open and close must look like 09:00")
    return cafe.get_hours()


@app.get("/api/cafe/availability")
def cafe_availability(date: str | None = None, time: str | None = None, party: int = 2, kind: str | None = None):
    start = None
    if time:
        try:
            start = cafe._hhmm(time)
        except ValueError:
            raise HTTPException(status_code=400, detail="time must look like 19:30")
    return cafe.availability(_day(date), start, party, kind if kind in cafe.KINDS else None)


@app.get("/api/cafe/bookings")
def cafe_bookings(date: str | None = None):
    return cafe.bookings_on(_day(date), include_cancelled=True)


@app.post("/api/cafe/bookings/{code}/cancel")
def cafe_cancel(code: str, x_admin_token: str | None = Header(None)):
    _admin(x_admin_token)
    if cafe.cancel_booking(code) is None:
        raise HTTPException(status_code=404, detail="No active booking with that code")
    return {"ok": True}


# ---------------------------------------------------------------- frontend
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
