"""SQLite store for per-turn metadata (user input, bot response, date/time, latency)."""
import csv
import io
import sqlite3
import statistics
import threading
from datetime import datetime

from . import config

FIELDS = [
    "turn_id", "session_id", "date", "time", "timezone", "timestamp_iso",
    "input_mode", "user_input", "bot_response",
    "stt_seconds", "llm_seconds", "tts_seconds",
    "server_total_seconds",          # request received -> reply + audio ready on server
    "client_time_to_audio_seconds",  # user stops speaking -> bot audio starts playing (browser)
    "playback_seconds",              # length of bot audio actually played (browser)
    "llm_first_sentence_seconds",    # request received -> first sentence ready from the LLM
    "tts_first_seconds",             # time to voice the first sentence
    "server_first_audio_seconds",    # request received -> first audio chunk ready on server
    "slowest_stage",                 # set only when the reply was slow: which step took longest
    "answer_function",               # which function produced the answer text
    "call_chain",                    # every function called for this turn, in order
    "tts_provider", "llm_model", "status", "error",
]
_REAL = {"stt_seconds", "llm_seconds", "tts_seconds", "server_total_seconds",
         "client_time_to_audio_seconds", "playback_seconds",
         "llm_first_sentence_seconds", "tts_first_seconds", "server_first_audio_seconds"}

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    cols = ", ".join(
        f"{f} REAL" if f in _REAL else f"{f} TEXT" + (" PRIMARY KEY" if f == "turn_id" else "")
        for f in FIELDS
    )
    with _lock, _connect() as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS turns ({cols})")
        existing = {r[1] for r in conn.execute("PRAGMA table_info(turns)")}
        for f in FIELDS:  # upgrade databases created by an older version
            if f not in existing:
                conn.execute(f"ALTER TABLE turns ADD COLUMN {f} {'REAL' if f in _REAL else 'TEXT'}")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)")


def now_parts() -> dict:
    now = datetime.now().astimezone()
    return {
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "UTC" + now.strftime("%z")[:3] + ":" + now.strftime("%z")[3:],  # e.g. UTC+05:30
        "timestamp_iso": now.isoformat(timespec="milliseconds"),
    }


def save_turn(record: dict) -> None:
    row = {f: record.get(f) for f in FIELDS}
    placeholders = ", ".join("?" for _ in FIELDS)
    with _lock, _connect() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO turns ({', '.join(FIELDS)}) VALUES ({placeholders})",
            [row[f] for f in FIELDS],
        )


def update_client_metrics(turn_id: str, time_to_audio: float | None, playback: float | None,
                          tts_first: float | None = None, tts_total: float | None = None) -> bool:
    # tts_first / tts_total are measured in the browser for "browser" voice (the server does no TTS then).
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE turns SET client_time_to_audio_seconds = ?, playback_seconds = ?, "
            "tts_first_seconds = COALESCE(?, tts_first_seconds), tts_seconds = COALESCE(?, tts_seconds) "
            "WHERE turn_id = ?",
            (time_to_audio, playback, tts_first, tts_total, turn_id),
        )
        return cur.rowcount > 0


def get_turn(turn_id: str) -> dict | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
    return dict(row) if row else None


def list_turns(limit: int = 50, session_id: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM turns", []
    if session_id:
        sql += " WHERE session_id = ?"
        args.append(session_id)
    with _lock, _connect() as conn:
        rows = conn.execute(sql + " ORDER BY timestamp_iso DESC LIMIT ?", (*args, limit)).fetchall()
    return [dict(r) for r in rows]


def to_csv(session_id: str | None = None) -> str:
    sql, args = "SELECT * FROM turns", []
    if session_id:
        sql += " WHERE session_id = ?"
        args.append(session_id)
    with _lock, _connect() as conn:
        rows = conn.execute(sql + " ORDER BY timestamp_iso", args).fetchall()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
    return buf.getvalue()


def stats() -> dict:
    """Latency summary using the best available 'total' per turn."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT server_total_seconds, client_time_to_audio_seconds FROM turns WHERE status IN ('success','empty_transcription','exit')"
        ).fetchall()
    server = [r[0] for r in rows if r[0] is not None]
    client = [r[1] for r in rows if r[1] is not None]

    def summarize(values: list[float]) -> dict | None:
        if not values:
            return None
        return {
            "count": len(values),
            "average": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "fastest": round(min(values), 3),
            "slowest": round(max(values), 3),
        }

    return {"server_total": summarize(server), "client_time_to_audio": summarize(client)}


# ---------------------------------------------------------------- history management
DEFAULT_TITLE = "New conversation"


def _title_from(text: str) -> str:
    text = " ".join((text or "").split())
    return (text[:48] + "…") if len(text) > 48 else text


def touch_conversation(conv_id: str, first_text: str, when_iso: str) -> None:
    """Create the conversation on its first turn (titled from the first question), else bump updated_at."""
    with _lock, _connect() as conn:
        row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, _title_from(first_text) or DEFAULT_TITLE, when_iso, when_iso),
            )
        else:
            title = row["title"]
            if title == DEFAULT_TITLE and first_text:
                title = _title_from(first_text)
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, when_iso, conv_id)
            )


def _like(q: str) -> str:
    return "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def list_conversations(q: str | None = None) -> list[dict]:
    sql = (
        "SELECT c.id, c.title, c.created_at, c.updated_at, "
        "(SELECT COUNT(*) FROM turns t WHERE t.session_id = c.id) AS turn_count "
        "FROM conversations c"
    )
    args: list = []
    if q and q.strip():
        like = _like(q.strip())
        sql += (
            " WHERE c.title LIKE ? ESCAPE '\\' OR EXISTS (SELECT 1 FROM turns t WHERE t.session_id = c.id "
            "AND (t.user_input LIKE ? ESCAPE '\\' OR t.bot_response LIKE ? ESCAPE '\\'))"
        )
        args = [like, like, like]
    with _lock, _connect() as conn:
        rows = conn.execute(sql + " ORDER BY c.updated_at DESC", args).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> dict | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if row is None:
            return None
        turns = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp_iso", (conv_id,)
        ).fetchall()
    return {**dict(row), "turns": [dict(t) for t in turns]}


def rename_conversation(conv_id: str, title: str) -> bool:
    title = " ".join(title.split())[:80] or DEFAULT_TITLE
    with _lock, _connect() as conn:
        return conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id)).rowcount > 0


def delete_conversation(conv_id: str) -> bool:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM turns WHERE session_id = ?", (conv_id,))
        return conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,)).rowcount > 0


def delete_all() -> int:
    with _lock, _connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        conn.execute("DELETE FROM turns")
        conn.execute("DELETE FROM conversations")
    return n


def llm_context(conv_id: str, max_messages: int) -> list[dict]:
    """Rebuild the LLM message history of a saved conversation so it can be continued."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT user_input, bot_response FROM turns WHERE session_id = ? "
            "AND status IN ('success', 'tts_error') ORDER BY timestamp_iso",
            (conv_id,),
        ).fetchall()
    messages: list[dict] = []
    for r in rows:
        messages.append({"role": "user", "content": r["user_input"]})
        messages.append({"role": "assistant", "content": r["bot_response"]})
    return messages[-max_messages:]
