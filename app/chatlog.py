"""Per-chat log files in logs/chats/.

For every conversation two files are (re)written after each turn:
  logs/chats/<start date>_<start time>_<id>.json   full details, machine readable
  logs/chats/<start date>_<start time>_<id>.txt    the same, easy to read
plus one summary row per chat in logs/chat_sessions.csv.

Stored: when the chat started, last activity, chat duration, every question and answer with its
date/time, which function answered, the call chain, and all timings (speech-to-text, LLM, voice,
server total, time to first sound, playback).
"""
import csv
import json
import re
import threading
from datetime import datetime

from . import config, metadata

_lock = threading.Lock()
SUMMARY_CSV = config.LOG_DIR / "chat_sessions.csv"
_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _id(session_id: str) -> str:
    return _SAFE.sub("", session_id)[:36] or "chat"


def _dt(iso: str | None):
    try:
        return datetime.fromisoformat(iso) if iso else None
    except ValueError:
        return None


def _num(v):
    return v if isinstance(v, (int, float)) else None


def _build(conv: dict) -> dict:
    turns = conv.get("turns") or []
    first = _dt(turns[0]["timestamp_iso"]) if turns else _dt(conv.get("created_at"))
    last_end, last_start = None, None
    rows, funcs, responses = [], {}, []
    for n, t in enumerate(turns, 1):
        start = _dt(t.get("timestamp_iso"))
        # a turn ends when the reply has been generated and (if known) fully played
        length = (_num(t.get("server_total_seconds")) or 0) + (_num(t.get("playback_seconds")) or 0)
        if start:
            from datetime import timedelta
            end = start + timedelta(seconds=length)
            last_end = end if last_end is None or end > last_end else last_end
            last_start = start
        resp = _num(t.get("client_time_to_audio_seconds"))
        if resp is None:
            resp = _num(t.get("server_first_audio_seconds"))
        if resp is not None:
            responses.append(resp)
        fn = t.get("answer_function") or "(none)"
        funcs[fn] = funcs.get(fn, 0) + 1
        rows.append({
            "turn": n,
            "date": t.get("date"), "time": t.get("time"), "timezone": t.get("timezone"),
            "input_mode": t.get("input_mode"),
            "user_input": t.get("user_input"), "bot_response": t.get("bot_response"),
            "answer_function": t.get("answer_function"), "slowest_stage_if_slow": t.get("slowest_stage") or "",
            "call_chain": t.get("call_chain"),
            "response_time_seconds": resp,  # what the user waited: stop speaking -> first sound
            "timings_seconds": {
                "speech_to_text": t.get("stt_seconds"),
                "llm_first_sentence": t.get("llm_first_sentence_seconds"),
                "llm_total": t.get("llm_seconds"),
                "voice_first_chunk": t.get("tts_first_seconds"),
                "voice_total_all_chunks": t.get("tts_seconds"),
                "server_first_audio": t.get("server_first_audio_seconds"),
                "server_total": t.get("server_total_seconds"),
                "client_time_to_first_sound": t.get("client_time_to_audio_seconds"),
                "bot_playback": t.get("playback_seconds"),
            },
            "tts_provider": t.get("tts_provider"), "llm_model": t.get("llm_model"),
            "status": t.get("status"), "error": t.get("error") or "",
        })
    duration = (last_end - first).total_seconds() if first and last_end else 0.0
    return {
        "session_id": conv["id"],
        "title": conv.get("title"),
        "chat_started": {
            "date": first.date().isoformat() if first else None,
            "time": first.strftime("%H:%M:%S") if first else None,
            "timestamp_iso": first.isoformat(timespec="milliseconds") if first else None,
        },
        "last_activity_iso": last_start.isoformat(timespec="milliseconds") if last_start else None,
        "chat_ended_iso": last_end.isoformat(timespec="milliseconds") if last_end else None,
        "chat_duration_seconds": round(duration, 1),
        "turn_count": len(rows),
        "response_time_seconds": {
            "average": round(sum(responses) / len(responses), 3) if responses else None,
            "fastest": round(min(responses), 3) if responses else None,
            "slowest": round(max(responses), 3) if responses else None,
            "over_1_5s": sum(1 for r in responses if r > 1.5),
        },
        "functions_used": funcs,
        "errors": sum(1 for r in rows if r["status"] not in ("success", "exit")),
        "turns": rows,
    }


def _text(c: dict) -> str:
    s = c["chat_started"]
    rt = c["response_time_seconds"]
    out = [
        f"CHAT: {c['title']}",
        f"Session id      : {c['session_id']}",
        f"Started         : {s['date']} {s['time']}",
        f"Ended           : {c['chat_ended_iso']}",
        f"Duration        : {c['chat_duration_seconds']} s",
        f"Turns           : {c['turn_count']}",
        f"Response time   : avg {rt['average']} s, fastest {rt['fastest']} s, slowest {rt['slowest']} s,"
        f" {rt['over_1_5s']} over 1.5 s",
        f"Functions used  : {', '.join(f'{k} x{v}' for k, v in c['functions_used'].items())}",
        "=" * 78,
    ]
    for t in c["turns"]:
        tm = t["timings_seconds"]
        out += [
            f"[{t['turn']}] {t['date']} {t['time']} {t['timezone']}  ({t['input_mode']})",
            f"  You    : {t['user_input']}",
            f"  Bot    : {t['bot_response']}",
            f"  Answered by : {t['answer_function']}   status={t['status']}"
            + (f"   error={t['error']}" if t["error"] else ""),
            f"  Response time (to first sound): {t['response_time_seconds']} s"
            + (f"   SLOW: {t['slowest_stage_if_slow']} took longest" if t["slowest_stage_if_slow"] else ""),
            "  Timings: " + ", ".join(f"{k}={v}" for k, v in tm.items() if v is not None),
            f"  Voice: {t['tts_provider']}   LLM: {t['llm_model']}",
            f"  Call chain: {t['call_chain']}",
            "-" * 78,
        ]
    return "\n".join(out) + "\n"


def _paths(session_id: str):
    return sorted(config.CHAT_LOG_DIR.glob(f"*_{_id(session_id)}.json"))


def write(session_id: str) -> None:
    """(Re)write this chat's log files and the summary CSV. Never raises: logging must not break a reply."""
    try:
        conv = metadata.get_conversation(session_id)
        if conv is None:
            return
        c = _build(conv)
        stamp = f"{c['chat_started']['date']}_{(c['chat_started']['time'] or '').replace(':', '-')}"
        base = config.CHAT_LOG_DIR / f"{stamp}_{_id(session_id)}"
        with _lock:
            base.with_suffix(".json").write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
            base.with_suffix(".txt").write_text(_text(c), encoding="utf-8")
            _write_summary()
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger("voicebot").exception("Could not write chat log")


def _write_summary() -> None:
    cols = ["chat_started_date", "chat_started_time", "chat_ended", "chat_duration_seconds", "turns",
            "avg_response_seconds", "fastest_response_seconds", "slowest_response_seconds",
            "replies_over_1_5s", "functions_used", "errors", "title", "session_id"]
    rows = []
    for f in sorted(config.CHAT_LOG_DIR.glob("*.json")):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        rt = c["response_time_seconds"]
        rows.append([c["chat_started"]["date"], c["chat_started"]["time"], c["chat_ended_iso"],
                     c["chat_duration_seconds"], c["turn_count"], rt["average"], rt["fastest"],
                     rt["slowest"], rt["over_1_5s"],
                     "; ".join(f"{k} x{v}" for k, v in c["functions_used"].items()),
                     c["errors"], c["title"], c["session_id"]])
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)


def delete(session_id: str) -> None:
    with _lock:
        for p in _paths(session_id):
            p.unlink(missing_ok=True)
            p.with_suffix(".txt").unlink(missing_ok=True)
        _write_summary()


def delete_all() -> None:
    with _lock:
        for p in config.CHAT_LOG_DIR.glob("*"):
            if p.suffix in (".json", ".txt"):
                p.unlink(missing_ok=True)
        _write_summary()
