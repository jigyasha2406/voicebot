"""Cafe seat booking: the seat / opening-hours / bookings database and the voice booking dialogue.

Two parts:

1. Database (MySQL, via PyMySQL): ``seats`` (every table and its type + capacity),
   ``opening_hours`` (one row per weekday) and ``bookings``. Availability is computed from these
   tables, never guessed by the LLM. The database and tables are created on first start.
2. Dialogue (``handle``): a small state machine that asks, one question at a time,
   special or normal table -> number of people -> day -> time -> name -> confirmation, then writes
   the booking. It understands English, Hindi (Devanagari) and Hinglish and fills several answers
   from one sentence ("normal table for four tomorrow at 7 pm"). The caller can also ask for one
   particular table ("Table 5", "Window 2"); if that table is already booked at that time the bot says so
   and asks for a different table and time slot. Anything that is not about booking
   returns ``None`` so the normal LLM answers it, exactly as before.
"""
from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pymysql
from pymysql.cursors import DictCursor

from . import config

_lock = threading.RLock()
KINDS = ("normal", "special")

# ------------------------------------------------------------------ database
SEED_SEATS = [  # (id, name, kind, capacity)
    ("N1", "Table 1", "normal", 2), ("N2", "Table 2", "normal", 2), ("N3", "Table 3", "normal", 2),
    ("N4", "Table 4", "normal", 4), ("N5", "Table 5", "normal", 4), ("N6", "Table 6", "normal", 4),
    ("N7", "Table 7", "normal", 6), ("N8", "Table 8", "normal", 6),
    ("S1", "Window 1", "special", 2), ("S2", "Window 2", "special", 2),
    ("S3", "Garden 1", "special", 4), ("S4", "Garden 2", "special", 4),
    ("S5", "Lounge 1", "special", 6), ("S6", "Private room", "special", 10),
]

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS seats (
        id VARCHAR(16) PRIMARY KEY,
        name VARCHAR(64) NOT NULL,
        kind ENUM('normal','special') NOT NULL,
        capacity INT NOT NULL,
        active TINYINT(1) NOT NULL DEFAULT 1
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS opening_hours (
        weekday TINYINT PRIMARY KEY,              -- 0 = Monday
        open_min INT NOT NULL,
        close_min INT NOT NULL,
        closed TINYINT(1) NOT NULL DEFAULT 0
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS bookings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        code CHAR(4) NOT NULL,
        session_id VARCHAR(64),
        name VARCHAR(100),
        party INT,
        kind ENUM('normal','special'),
        `date` DATE NOT NULL,
        start_min INT NOT NULL,
        end_min INT NOT NULL,
        seat_ids VARCHAR(100) NOT NULL,           -- comma separated seat ids
        status ENUM('confirmed','cancelled') NOT NULL DEFAULT 'confirmed',
        created_at DATETIME,
        INDEX idx_bookings_date (`date`, status),
        INDEX idx_bookings_code (code, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


class _Conn:
    """Thin wrapper so callers can write ``conn.execute(sql, args)`` and iterate / fetch the result."""

    def __init__(self, raw: pymysql.connections.Connection):
        self.raw = raw

    def execute(self, sql: str, args=None):
        cur = self.raw.cursor()
        cur.execute(sql, args)
        return cur

    def close(self) -> None:
        try:
            self.raw.close()
        except Exception:
            pass


def _clean(row: dict | None) -> dict | None:
    """MySQL returns DATE / DATETIME objects; the rest of the app works with ISO strings."""
    if row is None:
        return None
    return {k: (v.isoformat() if isinstance(v, (date, datetime)) else v) for k, v in row.items()}


def _connect(with_db: bool = True) -> _Conn:
    kwargs = dict(host=config.MYSQL_HOST, port=config.MYSQL_PORT, user=config.MYSQL_USER,
                  password=config.MYSQL_PASSWORD, charset="utf8mb4", cursorclass=DictCursor,
                  autocommit=True, connect_timeout=10)
    if with_db:
        kwargs["database"] = config.MYSQL_DATABASE
    return _Conn(pymysql.connect(**kwargs))


def _rows(cur) -> list[dict]:
    return [_clean(r) for r in cur.fetchall()]


def _hhmm(text: str) -> int:
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def init_db() -> None:
    """Create the database and tables if they do not exist, and seed seats / opening hours once."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", config.MYSQL_DATABASE):
        raise RuntimeError("MYSQL_DATABASE may only contain letters, digits and underscores")
    with _lock:
        try:
            conn = _connect(with_db=False)
        except pymysql.err.OperationalError as exc:
            raise RuntimeError(
                f"Cannot connect to MySQL at {config.MYSQL_HOST}:{config.MYSQL_PORT} as '{config.MYSQL_USER}'. "
                "Start the MySQL server and check MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD in .env "
                f"({exc})") from exc
        try:
            conn.execute(f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DATABASE}` "
                         "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            conn.execute(f"USE `{config.MYSQL_DATABASE}`")
            for ddl in SCHEMA:
                conn.execute(ddl)
            if conn.execute("SELECT COUNT(*) AS n FROM seats").fetchone()["n"] == 0:
                for s in SEED_SEATS:
                    conn.execute("INSERT INTO seats (id, name, kind, capacity) VALUES (%s,%s,%s,%s)", s)
            if conn.execute("SELECT COUNT(*) AS n FROM opening_hours").fetchone()["n"] == 0:
                o, c = _hhmm(config.CAFE_OPEN), _hhmm(config.CAFE_CLOSE)
                for d in range(7):
                    conn.execute("INSERT INTO opening_hours (weekday, open_min, close_min, closed) VALUES (%s,%s,%s,0)",
                                 (d, o, c))
        finally:
            conn.close()


def now() -> datetime:
    """Current time in the cafe's timezone as a naive datetime."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(config.CAFE_TZ)).replace(tzinfo=None)
    except Exception:  # tzdata missing (common on Windows): fall back to the server clock
        return datetime.now()


def hours_for(d: date) -> tuple[int, int] | None:
    """(open_min, close_min) for that day, or None when the cafe is closed."""
    with _lock:
        conn = _connect()
        try:
            r = conn.execute("SELECT * FROM opening_hours WHERE weekday = %s", (d.weekday(),)).fetchone()
        finally:
            conn.close()
    if r is None or r["closed"]:
        return None
    return r["open_min"], r["close_min"]


def _free_seats(conn: _Conn, d: date, start: int, end: int, kind: str | None) -> list[dict]:
    busy: set[str] = set()
    cur = conn.execute("SELECT seat_ids FROM bookings WHERE `date` = %s AND status = 'confirmed' "
                       "AND start_min < %s AND end_min > %s", (d.isoformat(), end, start))
    for b in cur.fetchall():
        busy.update(b["seat_ids"].split(","))
    sql, args = "SELECT * FROM seats WHERE active = 1", []
    if kind:
        sql += " AND kind = %s"
        args.append(kind)
    return [s for s in conn.execute(sql + " ORDER BY capacity, id", args).fetchall() if s["id"] not in busy]


def _allocate(free: list[dict], party: int, seat_id: str | None = None) -> list[dict] | None:
    """The wanted table if one was asked for; else the smallest single table that fits;
    otherwise join up to 3 free tables (largest first)."""
    if seat_id:
        return [s for s in free if s["id"] == seat_id and s["capacity"] >= party][:1] or None
    fits = [s for s in free if s["capacity"] >= party]
    if fits:
        return [fits[0]]
    chosen, total = [], 0
    for s in sorted(free, key=lambda s: -s["capacity"]):
        chosen.append(s)
        total += s["capacity"]
        if total >= party:
            return chosen if len(chosen) <= 3 else None
        if len(chosen) == 3:
            return None
    return None


def _end_for(start: int, close: int) -> int:
    return min(start + config.BOOKING_MINUTES, close)


def check_slot(d: date, start: int, party: int, kind: str, seat_id: str | None = None) -> list[dict] | None:
    hrs = hours_for(d)
    if hrs is None:
        return None
    with _lock:
        conn = _connect()
        try:
            return _allocate(_free_seats(conn, d, start, _end_for(start, hrs[1]), kind), party, seat_id)
        finally:
            conn.close()


def free_seat_names(d: date, start: int, party: int, kind: str | None, limit: int = 3) -> list[str]:
    """Names of single tables that are free then and big enough (used to suggest a different table)."""
    hrs = hours_for(d)
    if hrs is None:
        return []
    with _lock:
        conn = _connect()
        try:
            free = _free_seats(conn, d, start, _end_for(start, hrs[1]), kind)
        finally:
            conn.close()
    return [s["name"] for s in free if s["capacity"] >= party][:limit]


def nearby_times(d: date, start: int, party: int, kind: str, limit: int = 3, seat_id: str | None = None) -> list[int]:
    """Free start times on that day (every SLOT_STEP minutes), nearest to the wanted time first."""
    hrs = hours_for(d)
    if hrs is None:
        return []
    earliest = 0
    if d == now().date():
        n = now()
        earliest = n.hour * 60 + n.minute + config.MIN_LEAD_MINUTES
    found = []
    t = hrs[0]
    while t <= hrs[1] - config.MIN_STAY_MINUTES:
        if t >= earliest and check_slot(d, t, party, kind, seat_id):
            found.append(t)
        t += config.SLOT_STEP
    found.sort(key=lambda x: abs(x - start))
    return sorted(found[:limit])


def create_booking(session_id: str, name: str, party: int, kind: str, d: date, start: int,
                   seat_id: str | None = None) -> dict | None:
    """Atomically re-check availability and write the booking. None if the table/slot was just taken.

    The seats of that type are locked (SELECT ... FOR UPDATE) inside a transaction, so two callers
    (even in two server processes) can never be given the same table at overlapping times."""
    hrs = hours_for(d)
    if hrs is None:
        return None
    end = _end_for(start, hrs[1])
    with _lock:
        conn = _connect()
        try:
            conn.execute("START TRANSACTION")
            try:
                conn.execute("SELECT id FROM seats WHERE kind = %s FOR UPDATE", (kind,)).fetchall()
                seats = _allocate(_free_seats(conn, d, start, end, kind), party, seat_id)
                if not seats:
                    conn.execute("ROLLBACK")
                    return None
                used = {r["code"] for r in conn.execute("SELECT code FROM bookings").fetchall()}
                code = next(c for c in (f"{random.randint(1000, 9999)}" for _ in range(500)) if c not in used)
                rec = {"code": code, "session_id": session_id, "name": name, "party": party, "kind": kind,
                       "date": d.isoformat(), "start_min": start, "end_min": end,
                       "seat_ids": ",".join(s["id"] for s in seats), "status": "confirmed",
                       "created_at": now().isoformat(timespec="seconds", sep=" ")}
                conn.execute("INSERT INTO bookings (code, session_id, name, party, kind, `date`, start_min, end_min, "
                             "seat_ids, status, created_at) VALUES (%(code)s,%(session_id)s,%(name)s,%(party)s,"
                             "%(kind)s,%(date)s,%(start_min)s,%(end_min)s,%(seat_ids)s,%(status)s,%(created_at)s)", rec)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            rec["seat_names"] = [s["name"] for s in seats]
            return rec
        finally:
            conn.close()


def cancel_booking(code: str) -> dict | None:
    with _lock:
        conn = _connect()
        try:
            r = _clean(conn.execute("SELECT * FROM bookings WHERE code = %s AND status = 'confirmed'",
                                    (code,)).fetchone())
            if r is None:
                return None
            conn.execute("UPDATE bookings SET status = 'cancelled' WHERE id = %s", (r["id"],))
            return r
        finally:
            conn.close()


# ---- read / admin helpers used by the REST API and the page
def list_seats() -> list[dict]:
    with _lock:
        conn = _connect()
        try:
            return _rows(conn.execute("SELECT * FROM seats ORDER BY kind DESC, id"))
        finally:
            conn.close()


def get_seat(seat_id: str) -> dict | None:
    with _lock:
        conn = _connect()
        try:
            return _clean(conn.execute("SELECT * FROM seats WHERE id = %s", (seat_id,)).fetchone())
        finally:
            conn.close()


def add_seat(seat_id: str, name: str, kind: str, capacity: int) -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute("INSERT INTO seats (id, name, kind, capacity) VALUES (%s,%s,%s,%s)",
                         (seat_id, name, kind, capacity))
        finally:
            conn.close()


def update_seat(seat_id: str, **fields) -> bool:
    allowed = {k: v for k, v in fields.items() if k in ("name", "kind", "capacity", "active") and v is not None}
    if not allowed:
        return False
    with _lock:
        conn = _connect()
        try:
            # matched rows (not changed rows) decide "found", so setting a value to what it already is still succeeds
            n = conn.execute("SELECT COUNT(*) AS n FROM seats WHERE id = %s", (seat_id,)).fetchone()["n"]
            if not n:
                return False
            conn.execute(f"UPDATE seats SET {', '.join(f'{k} = %s' for k in allowed)} WHERE id = %s",
                         (*allowed.values(), seat_id))
            return True
        finally:
            conn.close()


def get_hours() -> list[dict]:
    with _lock:
        conn = _connect()
        try:
            return [{"weekday": r["weekday"], "open": f"{r['open_min'] // 60:02d}:{r['open_min'] % 60:02d}",
                     "close": f"{r['close_min'] // 60:02d}:{r['close_min'] % 60:02d}", "closed": bool(r["closed"])}
                    for r in _rows(conn.execute("SELECT * FROM opening_hours ORDER BY weekday"))]
        finally:
            conn.close()


def set_hours(weekday: int, open_t: str, close_t: str, closed: bool) -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute("INSERT INTO opening_hours (weekday, open_min, close_min, closed) VALUES (%s,%s,%s,%s) "
                         "ON DUPLICATE KEY UPDATE open_min = VALUES(open_min), close_min = VALUES(close_min), "
                         "closed = VALUES(closed)",
                         (weekday, _hhmm(open_t), _hhmm(close_t), int(closed)))
        finally:
            conn.close()


def bookings_on(d: date, include_cancelled: bool = False) -> list[dict]:
    sql = "SELECT * FROM bookings WHERE `date` = %s" + ("" if include_cancelled else " AND status = 'confirmed'")
    with _lock:
        conn = _connect()
        try:
            return _rows(conn.execute(sql + " ORDER BY start_min, code", (d.isoformat(),)))
        finally:
            conn.close()


def availability(d: date, start: int | None, party: int, kind: str | None) -> dict:
    """Which seats are free on a day (at a given time, or any time) for the REST API."""
    hrs = hours_for(d)
    if hrs is None:
        return {"date": d.isoformat(), "closed": True, "seats": []}
    s = hrs[0] if start is None else start
    with _lock:
        conn = _connect()
        try:
            free = _free_seats(conn, d, s, _end_for(s, hrs[1]), kind)
        finally:
            conn.close()
    return {"date": d.isoformat(), "closed": False, "time": fmt_clock(s), "party": party,
            "free_seats": [dict(x) for x in free], "can_seat_party": bool(_allocate(free, party))}


def overview(d: date) -> dict:
    seats, books = list_seats(), bookings_on(d)
    for s in seats:
        s["bookings"] = [{"code": b["code"], "name": b["name"], "party": b["party"],
                          "from": fmt_clock(b["start_min"]), "to": fmt_clock(b["end_min"])}
                         for b in books if s["id"] in b["seat_ids"].split(",")]
    hrs = hours_for(d)
    return {"cafe": config.CAFE_NAME, "date": d.isoformat(), "weekday": d.strftime("%A"),
            "open": fmt_clock(hrs[0]) if hrs else None, "close": fmt_clock(hrs[1]) if hrs else None,
            "closed": hrs is None, "seats": seats, "bookings": books}


def fmt_clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ------------------------------------------------------------------ spoken formatting
HI_DAYS = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


def say_time(minutes: int, lang: str) -> str:
    h, m = divmod(minutes, 60)
    h12 = h % 12 or 12
    clock = f"{h12}" if m == 0 else f"{h12}:{m:02d}"
    if lang == "hi":
        part = "सुबह" if h < 12 else "दोपहर" if h < 16 else "शाम" if h < 20 else "रात"
        return f"{part} {clock} बजे"
    return f"{clock} {'AM' if h < 12 else 'PM'}"


def say_date(d: date, lang: str) -> str:
    today = now().date()
    if d == today:
        return "आज" if lang == "hi" else "today"
    if d == today + timedelta(days=1):
        return "कल" if lang == "hi" else "tomorrow"
    if lang == "hi":
        return f"{HI_DAYS[d.weekday()]} {d.day} {HI_MONTHS[d.month - 1]}"
    return f"{d.strftime('%A')}, {d.day} {d.strftime('%B')}"


def say_kind(kind: str, lang: str) -> str:
    return {"normal": "नॉर्मल", "special": "स्पेशल"}[kind] if lang == "hi" else kind


def say_hours(d: date, lang: str) -> str:
    hrs = hours_for(d)
    if hrs is None:
        return "हम उस दिन बंद रहते हैं" if lang == "hi" else "we are closed that day"
    if lang == "hi":
        return f"{say_time(hrs[0], 'hi').replace(' बजे', '')} बजे से {say_time(hrs[1], 'hi').replace(' बजे', '')} बजे तक"
    return f"{say_time(hrs[0], 'en')} to {say_time(hrs[1], 'en')}"


_HI_SEAT = {"Table": "टेबल", "Window": "विंडो", "Garden": "गार्डन", "Lounge": "लाउंज", "Private room": "प्राइवेट रूम"}


def _seat_say(name: str, lang: str) -> str:
    if lang != "hi":
        return name
    for en, hi in _HI_SEAT.items():
        if name.startswith(en):
            return hi + name[len(en):]
    return name


def say_code(code: str) -> str:
    return " ".join(code)


def system_prompt() -> str:
    today = now().date()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hours = "; ".join(f"{days[h['weekday']]} " + ("closed" if h["closed"] else f"{h['open']} to {h['close']}")
                      for h in get_hours())
    seats = list_seats()
    nn = sum(1 for s in seats if s["kind"] == "normal" and s["active"])
    ns = sum(1 for s in seats if s["kind"] == "special" and s["active"])
    return (
        f"You are the voice assistant of {config.CAFE_NAME}, a cafe. "
        "Understand Hindi, English and Hinglish. Reply in the same language the user uses. "
        "Answer in one or two short sentences, spoken style, unless the user asks for detail: "
        "no markdown, no bullet points, no emojis. "
        f"Opening hours: {hours}. The cafe has {nn} normal tables and {ns} special tables "
        "(window, garden, lounge and a private room, good for birthdays and celebrations). "
        "You take table bookings by voice. When the user wants to book a table, tell them to say "
        "'book a table' and the booking system will ask for the details. "
        "Never invent availability, booking codes, prices or menu items; if you do not know something, say so honestly. "
        f"Today is {today.strftime('%A, %d %B %Y')}. You may also answer short general questions."
    )


# ------------------------------------------------------------------ language and number helpers
W = r"(?<![\w\u0900-\u097F])"   # word start (works for Devanagari too)
E = r"(?![\w\u0900-\u097F])"    # word end
_DEVA = re.compile(r"[\u0900-\u097F]")
_HINGLISH = re.compile(W + r"(haan|nahi|nahin|kal|aaj|parso|baje|kitne|mujhe|chahiye|hum|mera|naam|karo|kar do|"
                       r"batao|bukking|log|shaam|subah|raat)" + E, re.I)

NUMW = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "ek": 1, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6, "saat": 7, "aath": 8,
    "nau": 9, "das": 10, "gyarah": 11, "barah": 12, "pandrah": 15, "bees": 20,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17,
    "अठारह": 18, "उन्नीस": 19, "बीस": 20,
}
_NUM_ALT = "|".join(sorted(map(re.escape, NUMW), key=len, reverse=True))
NUM = rf"(\d{{1,2}}|{_NUM_ALT})"
_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4",
                "पांच": "5", "पाँच": "5", "छह": "6", "सात": "7", "आठ": "8", "नौ": "9"}


def _n(tok: str) -> int | None:
    tok = tok.lower()
    return int(tok) if tok.isdigit() else NUMW.get(tok)


def detect_lang(text: str, previous: str = "en") -> str:
    if _DEVA.search(text):
        return "hi"
    if _HINGLISH.search(text):
        return "hi"
    return previous if previous else "en"


def _cut(pattern: str, text: str, flags: int = re.I):
    """First regex match and the text with that match blanked out (so later parsers ignore it)."""
    m = re.search(pattern, text, flags)
    if not m:
        return None, text
    return m, text[:m.start()] + " " + text[m.end():]


# ------------------------------------------------------------------ parsers
_SPECIAL = re.compile(W + r"(special|premium|vip|window|garden|lounge|private|birthday|anniversary|celebration|"
                      r"romantic|candle ?light|स्पेशल|खास|विशेष|प्रीमियम|बर्थडे|जन्मदिन|एनिवर्सरी|विंडो|गार्डन|लाउंज)" + E, re.I)
_NORMAL = re.compile(W + r"(normal|regular|general|simple|standard|ordinary|usual|नॉर्मल|नार्मल|सामान्य|रेगुलर|साधारण)" + E, re.I)
_NOT_SPECIAL = re.compile(r"(not|no|don'?t want|without)\s+(a\s+)?special|special\s+(nahi|nahin|नहीं)|स्पेशल\s+नहीं", re.I)


def parse_kind(t: str) -> str | None:
    if _NOT_SPECIAL.search(t):
        return "normal"
    s, n = _SPECIAL.search(t), _NORMAL.search(t)
    if s and not n:
        return "special"
    if n and not s:
        return "normal"
    if s and n:
        return "special" if s.start() > n.start() else "normal"  # the later word is usually the correction
    return None


_DAYPART_AM = re.compile(W + r"(morning|subah|सुबह|savere|सवेरे)" + E, re.I)
_DAYPART_PM = re.compile(W + r"(afternoon|evening|night|tonight|dopahar|shaam|raat|दोपहर|शाम|रात)" + E, re.I)
_MER = r"(am|pm)"


def _resolve_hour(h: int, m: int, mer: str | None, t: str, d: date | None) -> int | None:
    """Turn a spoken hour into minutes after midnight, using am/pm, a day-part word or opening hours."""
    if m > 59 or h > 24:
        return None
    if h >= 13:                                   # already 24-hour clock
        return (h % 24) * 60 + m
    if mer == "pm" or (mer is None and _DAYPART_PM.search(t)):
        h = h + 12 if h < 12 else h
    elif mer == "am" or (mer is None and _DAYPART_AM.search(t)):
        h = 0 if h == 12 else h
    else:                                         # no hint: pick the reading that falls inside opening hours
        hrs = hours_for(d) if d else None
        lo, hi = hrs or (_hhmm(config.CAFE_OPEN), _hhmm(config.CAFE_CLOSE))
        cands = [12] if h == 12 else ([h, h + 12] if h >= 9 else [h + 12, h])
        h = next((c for c in cands if lo <= c * 60 + m <= hi), cands[0])
    return h * 60 + m


def parse_time(t: str, bare_ok: bool, d: date | None) -> tuple[int | None, str]:
    t = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", t, flags=re.I)
    mer = lambda m, g: (m.group(g) or "").lower() or None
    m, rest = _cut(rf"{W}(\d{{1,2}})\s*[:.]\s*(\d{{2}})\s*{_MER}?{E}", t)
    if m:
        return _resolve_hour(int(m.group(1)), int(m.group(2)), mer(m, 3), t, d), rest
    m, rest = _cut(rf"{W}(\d{{1,2}})\s*{_MER}{E}", t)
    if m:
        return _resolve_hour(int(m.group(1)), 0, mer(m, 2), t, d), rest
    m, rest = _cut(rf"{W}(half past|quarter past)\s+{NUM}{E}", t)
    if m and _n(m.group(2)):
        return _resolve_hour(_n(m.group(2)), 30 if "half" in m.group(1).lower() else 15, None, t, d), rest
    m, rest = _cut(rf"{W}quarter to\s+{NUM}{E}", t)
    if m and _n(m.group(1)):
        return _resolve_hour(_n(m.group(1)) - 1, 45, None, t, d), rest
    m, rest = _cut(rf"{W}(साढ़े|साढे|sadhe|सवा|sawa|पौने|paune)\s*{NUM}(\s*(बजे|baje))?{E}", t)
    if m and _n(m.group(2)):
        k, h = m.group(1).lower(), _n(m.group(2))
        if k in ("साढ़े", "साढे", "sadhe"):
            return _resolve_hour(h, 30, None, t, d), rest
        if k in ("सवा", "sawa"):
            return _resolve_hour(h, 15, None, t, d), rest
        return _resolve_hour(h - 1, 45, None, t, d), rest
    m, rest = _cut(rf"{W}(डेढ़|डेढ|dedh)\s*(बजे|baje)?{E}", t)
    if m:
        return _resolve_hour(1, 30, None, t, d), rest
    m, rest = _cut(rf"{W}(ढाई|dhai)\s*(बजे|baje)?{E}", t)
    if m:
        return _resolve_hour(2, 30, None, t, d), rest
    m, rest = _cut(rf"{W}{NUM}\s*(thirty|forty[ -]?five|fifteen)\s*{_MER}?{E}", t)
    if m and _n(m.group(1)):
        mins = {"thirty": 30, "fifteen": 15}.get(m.group(2).lower().replace(" ", "").replace("-", ""), 45)
        return _resolve_hour(_n(m.group(1)), mins, mer(m, 3), t, d), rest
    m, rest = _cut(rf"{W}{NUM}\s*(baje|बजे|o'?clock|o clock){E}", t)
    if m and _n(m.group(1)):
        return _resolve_hour(_n(m.group(1)), 0, None, t, d), rest
    m, rest = _cut(rf"{W}(noon|midday|12 noon){E}", t)
    if m:
        return 12 * 60, rest
    if bare_ok:
        m, rest = _cut(rf"{W}{NUM}{E}", t)
        if m and _n(m.group(1)) is not None:
            return _resolve_hour(_n(m.group(1)), 0, None, t, d), rest
    return None, t


_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
             "somvar": 0, "सोमवार": 0, "mangalvar": 1, "मंगलवार": 1, "budhvar": 2, "बुधवार": 2, "guruvar": 3,
             "गुरुवार": 3, "brihaspativar": 3, "shukravar": 4, "शुक्रवार": 4, "shanivar": 5, "शनिवार": 5,
             "ravivar": 6, "itwar": 6, "रविवार": 6}
_MONTHS = {m: i for i, names in enumerate(
    [("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
     ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"),
     ("november", "nov"), ("december", "dec")], 1) for m in names}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))


def parse_date(t: str) -> tuple[date | None, str]:
    today = now().date()
    m, rest = _cut(rf"{W}(day after tomorrow|parso|परसों|परसो){E}", t)
    if m:
        return today + timedelta(days=2), rest
    m, rest = _cut(rf"{W}(tomorrow|kal|कल){E}", t)
    if m:
        return today + timedelta(days=1), rest
    m, rest = _cut(rf"{W}(today|tonight|aaj|आज){E}", t)
    if m:
        return today, rest
    m, rest = _cut(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), rest
        except ValueError:
            return None, t
    m, rest = _cut(rf"{W}(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT}){E}", t)
    day_s, mon_s = (m.group(1), m.group(2)) if m else (None, None)
    if not m:
        m, rest = _cut(rf"{W}({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?{E}", t)
        if m:
            day_s, mon_s = m.group(2), m.group(1)
    if m:
        try:
            d = date(today.year, _MONTHS[mon_s.lower()], int(day_s))
            return (d if d >= today else d.replace(year=today.year + 1)), rest
        except ValueError:
            return None, t
    m, rest = _cut(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)", t)
    if m:
        try:
            d = date(today.year, int(m.group(2)), int(m.group(1)))
            return (d if d >= today else d.replace(year=today.year + 1)), rest
        except ValueError:
            return None, t
    wd_alt = "|".join(sorted(map(re.escape, _WEEKDAYS), key=len, reverse=True))
    m, rest = _cut(rf"{W}(next\s+|this\s+|coming\s+)?({wd_alt}){E}", t)
    if m:
        delta = (_WEEKDAYS[m.group(2).lower()] - today.weekday()) % 7
        if delta == 0 and m.group(1) and m.group(1).strip().lower() == "next":
            delta = 7
        return today + timedelta(days=delta), rest
    return None, t


_PEOPLE = r"(people|persons?|guests?|pax|members?|adults?|friends|log|logon|लोग|लोगों|व्यक्ति|जन|सदस्य|seats?|सीट|सीटें)"


def parse_party(t: str, bare_ok: bool) -> int | None:
    if re.search(W + r"(just me|only me|myself|alone|solo|अकेला|अकेले|akela|akele)" + E, t, re.I):
        return 1
    if re.search(W + r"(a couple|couple|जोड़ा)" + E, t, re.I):
        return 2
    for pat in (rf"{W}{NUM}\s*{_PEOPLE}{E}",
                rf"{W}(?:for|of|party of|group of|we are|we're|total|hum|हम)\s+{NUM}{E}"):
        m = re.search(pat, t, re.I)
        if m and _n(m.group(1)) is not None:
            return _n(m.group(1))
    if bare_ok:
        m = re.search(rf"{W}{NUM}{E}", t, re.I)
        if m and _n(m.group(1)) is not None:
            return _n(m.group(1))
    return None


_YES = re.compile(W + r"(yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|correct|right|proceed|go ahead|book it|"
                  r"haan|han|haa|ji|theek hai|thik hai|bilkul|done|हाँ|हां|हा|जी|ठीक है|ठीक|बिल्कुल|कर दो|कर दीजिए|करो)" + E, re.I)
_NO = re.compile(W + r"(no|nope|nah|not|don'?t|dont|nahi|nahin|na|wrong|incorrect|change|नहीं|नही|ना|मत|गलत|बदल)" + E, re.I)
_NAME_PREFIX = re.compile(r"^(?:my name is|my name's|the name is|name is|name's|it's|it is|this is|i am|i'm|call me|"
                          r"booking under|book it under|under|mera naam|मेरा नाम|नाम|naam)\s+", re.I)
_NAME_SUFFIX = re.compile(r"\s+(?:hai|है|please|ji|जी|sir|madam)$", re.I)
_NAME_EXPLICIT = re.compile(r"(?:my name is|my name's|mera naam|मेरा नाम)\s+(.+)", re.I)
_NOT_NAMES = {"yes", "no", "ok", "okay", "haan", "nahi", "नहीं", "हाँ", "हां", "sure", "confirm"}


def parse_name(text: str, bare_ok: bool) -> str | None:
    t = text.strip()
    m = _NAME_EXPLICIT.search(t)
    if m:
        t = m.group(1)
    elif not bare_ok:
        return None
    t = re.sub(r"[^\w\u0900-\u097F\s'-]", " ", t)
    t = _NAME_PREFIX.sub("", " ".join(t.split()))
    t = _NAME_SUFFIX.sub("", t).strip()
    if re.search(W + r"(skip|anonymous|no name|कोई नाम नहीं)" + E, t, re.I):
        return "Guest"
    words = t.split()
    if not words or len(words) > 3 or len(t) > 30 or re.search(r"\d", t) or t.lower() in _NOT_NAMES:
        return None
    return t.title() if t.isascii() else t


def parse_code(text: str) -> str | None:
    t = text.lower()
    for w, dgt in _DIGIT_WORDS.items():
        t = re.sub(W + re.escape(w) + E, dgt, t)
    t = re.sub(r"(?<=\d)[\s,.-]+(?=\d)", "", t)
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", t)
    return m.group(1) if m else None


_SEAT_WORDS = {"table": "table|टेबल|मेज़|मेज", "window": "window|विंडो", "garden": "garden|गार्डन", "lounge": "lounge|लाउंज"}


def parse_seat(t: str) -> tuple[dict | None, str]:
    """Find a request for one particular table ("table 5", "window two", "private room") in the sentence.
    Returns (seat row or None, text with that phrase blanked out). Only active seats can be chosen."""
    seats = [s for s in list_seats() if s["active"]]
    for s in seats:
        base, _, num = s["name"].rpartition(" ")
        word = (base if num.isdigit() else s["name"]).lower()
        if word == "private room":
            pat = rf"{W}(private\s+room|प्राइवेट\s+रूम){E}"
        else:
            if not num.isdigit() or word not in _SEAT_WORDS:
                continue
            pat = rf"{W}({_SEAT_WORDS[word]})\s*(?:number\s+|no\.?\s+|नंबर\s+|#)?{NUM}{E}"
        m, rest = _cut(pat, t)
        if not m:
            continue
        if word == "private room":
            return s, rest
        if _n(m.group(2)) == int(num):
            return s, rest
    # "the lounge" / "garden table" with a single such seat: pick it only for lounge (unique names)
    lounges = [s for s in seats if s["name"].lower().startswith("lounge")]
    if len(lounges) == 1:
        m, rest = _cut(rf"{W}(lounge|लाउंज){E}", t)
        if m:
            return lounges[0], rest
    return None, t


# intents
_BOOKVERB = re.compile(W + r"(book|booking|reserve|reservation|बुक|बुकिंग|रिज़र्व|रिजर्व|आरक्षण|reserv)" + E, re.I)
_SEATWORD = re.compile(W + r"(table|tables|seat|seats|टेबल|सीट|सीटें|jagah|जगह)" + E, re.I)
_AVAIL = re.compile(W + r"(available|availability|vacant|empty|free|उपलब्ध|खाली|khali|hai kya|है क्या)" + E, re.I)
_HOURS = re.compile(W + r"(timing|timings|opening|closing|open|opens|close|closes|closed|hours|खुला|खुलता|खुलते|"
                    r"बंद|टाइमिंग|कब तक|kab tak|kitne baje)" + E, re.I)
_CANCEL_WORD = re.compile(W + r"(cancel|कैंसिल|रद्द)" + E, re.I)
_BOOKING_NOUN = re.compile(W + r"(booking|reservation|बुकिंग)" + E, re.I)
_ABORT = re.compile(r"(cancel|never ?mind|forget it|leave it|no thanks|रहने दो|छोड़ो|छोड़ दो|कैंसिल|रद्द|नहीं चाहिए|nahi chahiye|"
                    r"मत करो|not now)", re.I)
_QUESTION_START = re.compile(r"\s*(what|which|how|why|who|where|do you|does|is there|are there|tell me|kya|kaun|kitna|kaise|"
                             r"क्या|कौन|कितना|कितने|कैसे|क्यों)" + E, re.I)
_RESTART = re.compile(r"(start over|restart|from the beginning|from scratch|फिर से शुरू|दोबारा शुरू|शुरू से)", re.I)
_QUESTION = re.compile(r"(\?|" + W + r"(what|which|how|why|who|where|do you|does|is there|are there|can i|can you|tell me|"
                       r"kya|kaun|kitna|kitne|kaise|kyun|क्या|कौन|कितना|कितने|कैसे|क्यों)" + E + ")", re.I)


# ------------------------------------------------------------------ dialogue
STEPS = ("kind", "party", "date", "time", "name")


@dataclass
class Draft:
    active: bool = False
    step: str = "kind"               # kind, party, date, time, name, confirm, offer, cancel_code
    kind: str | None = None
    party: int | None = None
    day: date | None = None
    start: int | None = None
    name: str | None = None
    seat: str | None = None         # id of one particular table, if the caller asked for it
    lang: str = "en"
    fails: int = 0
    fresh: bool = True               # True until the first answer is processed


@dataclass
class Reply:
    text: str | None                  # None = let the LLM answer, then speak `suffix`
    fn: str
    suffix: str | None = None
    booking: dict | None = field(default=None)


_drafts: dict[str, Draft] = {}


def reset(session_id: str) -> None:
    _drafts.pop(session_id, None)


def is_active(session_id: str) -> bool:
    d = _drafts.get(session_id)
    return bool(d and d.active)


M = {
    "en": {
        "kind": "Would you like a special table, like window, garden or lounge seating, or a normal table?",
        "party": "How many people will be coming?",
        "date": "Which day would you like to come? You can say today, tomorrow, or a date.",
        "time": "What time should I book it for? We are open {hours} that day.",
        "name": "Could I have a name for the booking?",
        "time_short": "What time should I book it for?",
        "confirm": "Let me confirm: a {kind} table for {party} {people}, {date} at {time}, under the name {name}. Shall I book it?",
        "booked": "Done! Your {kind} booking is confirmed for {party} {people}, {date} at {time}. You have {seats}. "
                  "Your booking code is {code}. See you soon!",
        "offer": "Would you like me to book a table for you?",
        "retry": "Sorry, I did not catch that. ",
        "no_change": "No problem. What would you like to change: the table type, the number of people, the day, or the time? "
                     "Or say cancel to stop.",
        "aborted": "Okay, I have stopped the booking. Say book a table whenever you are ready.",
        "restart": "Sure, let us start again. ",
        "closed": "Sorry, we are closed on {weekday}. Please pick another day.",
        "outside": "On that day we are open {hours}. Please choose a time in that window.",
        "past": "That time has already passed. Please choose a later time.",
        "toofar": "I can only book up to {n} days ahead. Please choose an earlier day.",
        "toolarge": "For groups larger than {n}, please call the cafe directly. How many people, if fewer?",
        "pastday": "That day has already passed. Which day would you like?",
        "alts": "Sorry, the {kind} table for {party} is already booked at {time} {date}. Please select a different table or time slot. "
                "The nearest free times are {alts}. Which one would you like?",
        "seat_booked": "Sorry, {seat} is already booked {date} at {time}. Please select a different table and time slot.",
        "seat_free": " Free at that time: {tables}. Or say another time for {seat}.",
        "seat_times": " {seat} is free at {alts}.",
        "seat_small": "Sorry, {seat} seats only {cap}. Please select a bigger table.",
        "seat_confirm": "Let me confirm: {seat} for {party} {people}, {date} at {time}, under the name {name}. Shall I book it?",
        "other": "Sorry, no {kind} table is free for {party} {date}, but a {other} table is available. Would you like a {other} table?",
        "full": "Sorry, we are fully booked {date} for {party} {people}. Please try another day.",
        "taken": "Sorry, that table was just booked by someone else, so it is already booked. Please select a different table and time slot. ",
        "avail_yes": "Yes, {date} at {time} we have {n_normal} normal and {n_special} special tables free. Would you like to book one?",
        "avail_no": "Sorry, nothing is free {date} at {time} for that group. Would you like me to find another time?",
        "avail_today": "We are open {hours}. Say the day and time you have in mind and I will check, or say book a table.",
        "hours": "We are open {hours} {date}.",
        "cancel_ask": "Sure. Please tell me your four digit booking code.",
        "cancel_ok": "Done. Booking {code} for {name} {date} at {time} is cancelled.",
        "cancel_bad": "I could not find an active booking with that code. Please say the four digit code again, or say never mind.",
        "voice_only": "Bookings are taken by voice only. Please tap the microphone and say it out loud.",
        "person": "person", "people": "people",
    },
    "hi": {
        "kind": "आप स्पेशल टेबल लेना चाहेंगे, जैसे विंडो, गार्डन या लाउंज, या नॉर्मल टेबल?",
        "party": "कितने लोग आएंगे?",
        "date": "आप किस दिन आना चाहेंगे? आज, कल या कोई तारीख बता दीजिए।",
        "time": "किस समय की बुकिंग करूँ? हम उस दिन {hours} खुले हैं।",
        "name": "बुकिंग किस नाम से करूँ?",
        "time_short": "किस समय की बुकिंग करूँ?",
        "confirm": "ठीक है, {kind} टेबल, {party} लोगों के लिए, {date} {time}, नाम {name}। क्या मैं बुक कर दूँ?",
        "booked": "हो गया! आपकी {kind} बुकिंग {party} लोगों के लिए {date} {time} कन्फर्म है। आपके लिए {seats}। "
                  "आपका बुकिंग कोड है {code}। आपका इंतज़ार रहेगा!",
        "offer": "क्या मैं आपके लिए टेबल बुक कर दूँ?",
        "retry": "माफ़ कीजिए, मैं समझ नहीं पाया। ",
        "no_change": "कोई बात नहीं। क्या बदलना है: टेबल का प्रकार, लोगों की संख्या, दिन या समय? या रद्द करने के लिए कैंसिल बोलिए।",
        "aborted": "ठीक है, बुकिंग रोक दी है। जब चाहें टेबल बुक करने के लिए बोल दीजिए।",
        "restart": "ठीक है, फिर से शुरू करते हैं। ",
        "closed": "माफ़ कीजिए, हम {weekday} को बंद रहते हैं। कृपया कोई और दिन बताइए।",
        "outside": "उस दिन हम {hours} खुले हैं। कृपया इसी समय के बीच का समय बताइए।",
        "past": "वह समय निकल चुका है। कृपया बाद का समय बताइए।",
        "toofar": "मैं सिर्फ {n} दिन आगे तक बुक कर सकता हूँ। कृपया पहले का दिन बताइए।",
        "toolarge": "{n} से ज़्यादा लोगों के लिए कृपया कैफ़े को सीधे फ़ोन कीजिए। अगर कम लोग हैं तो कितने?",
        "pastday": "वह दिन निकल चुका है। आप किस दिन आना चाहेंगे?",
        "alts": "माफ़ कीजिए, {date} {time} {party} लोगों के लिए {kind} टेबल पहले से बुक है। कृपया कोई दूसरी टेबल या दूसरा समय चुनिए। सबसे पास के खाली समय हैं {alts}। कौन सा चाहिए?",
        "seat_booked": "माफ़ कीजिए, {seat} {date} {time} पहले से बुक है। कृपया कोई दूसरी टेबल और दूसरा समय चुनिए।",
        "seat_free": " उस समय खाली हैं: {tables}। या {seat} के लिए कोई और समय बताइए।",
        "seat_times": " {seat} इन समय खाली है: {alts}।",
        "seat_small": "माफ़ कीजिए, {seat} में सिर्फ़ {cap} लोग बैठ सकते हैं। कृपया बड़ी टेबल चुनिए।",
        "seat_confirm": "ठीक है, {seat}, {party} लोगों के लिए, {date} {time}, नाम {name}। क्या मैं बुक कर दूँ?",
        "other": "माफ़ कीजिए, {date} {party} लोगों के लिए {kind} टेबल खाली नहीं है, लेकिन {other} टेबल मिल सकती है। क्या {other} टेबल चलेगी?",
        "full": "माफ़ कीजिए, {date} {party} लोगों के लिए सब बुक है। कृपया कोई और दिन बताइए।",
        "taken": "माफ़ कीजिए, वह टेबल अभी-अभी बुक हो गई, यानी पहले से बुक है। कृपया कोई दूसरी टेबल और दूसरा समय चुनिए। ",
        "avail_yes": "जी, {date} {time} हमारे पास {n_normal} नॉर्मल और {n_special} स्पेशल टेबल खाली हैं। क्या बुक कर दूँ?",
        "avail_no": "माफ़ कीजिए, {date} {time} इतने लोगों के लिए कुछ खाली नहीं है। क्या कोई और समय देखूँ?",
        "avail_today": "हम {hours} खुले हैं। दिन और समय बताइए, मैं देख लेता हूँ, या टेबल बुक करने को कहिए।",
        "hours": "हम {date} {hours} खुले हैं।",
        "cancel_ask": "ज़रूर। कृपया अपना चार अंकों का बुकिंग कोड बताइए।",
        "cancel_ok": "हो गया। {name} की बुकिंग {code}, {date} {time}, रद्द कर दी गई है।",
        "cancel_bad": "इस कोड की कोई बुकिंग नहीं मिली। कृपया चार अंकों का कोड फिर से बताइए, या रहने दीजिए बोलिए।",
        "voice_only": "बुकिंग सिर्फ़ आवाज़ से होती है। कृपया माइक्रोफ़ोन दबाकर बोलिए।",
        "person": "व्यक्ति", "people": "लोग",
    },
}


def _say(lang: str, key: str, **kw) -> str:
    return M[lang][key].format(**kw)


def _prompt(d: Draft) -> str:
    """The question for the first thing still missing (or the confirmation). Also records the step."""
    L = d.lang
    for step in STEPS:
        if getattr(d, {"date": "day", "time": "start"}.get(step, step)) is None:
            d.step = step
            if step == "time":
                return _say(L, "time", hours=say_hours(d.day, L)) if d.day else _say(L, "time_short")
            return _say(L, step)
    d.step = "confirm"
    if d.seat and (seat := get_seat(d.seat)):
        return _say(L, "seat_confirm", seat=_seat_say(seat["name"], L), party=d.party, people=_people(d.party, L),
                    date=say_date(d.day, L), time=say_time(d.start, L), name=d.name)
    return _say(L, "confirm", kind=say_kind(d.kind, L), party=d.party, people=_people(d.party, L),
                date=say_date(d.day, L), time=say_time(d.start, L), name=d.name)


def _people(n: int, lang: str) -> str:
    return M[lang]["person" if n == 1 else "people"]


def _validate(d: Draft) -> str | None:
    """Check the answers collected so far. Returns a spoken problem (and clears the bad answer) or None."""
    L, today = d.lang, now().date()
    if d.seat and d.party is not None:
        seat = get_seat(d.seat)
        if seat and d.party > seat["capacity"]:
            d.seat, d.step = None, "kind"
            return _say(L, "seat_small", seat=_seat_say(seat["name"], L), cap=seat["capacity"])
    if d.party is not None and d.party > config.MAX_PARTY:
        d.party, d.step = None, "party"
        return _say(L, "toolarge", n=config.MAX_PARTY)
    if d.day is not None:
        if d.day < today:
            d.day, d.start, d.step = None, None, "date"
            return _say(L, "pastday")
        if (d.day - today).days > config.MAX_DAYS_AHEAD:
            d.day, d.start, d.step = None, None, "date"
            return _say(L, "toofar", n=config.MAX_DAYS_AHEAD)
        hrs = hours_for(d.day)
        if hrs is None:
            wd = HI_DAYS[d.day.weekday()] if L == "hi" else d.day.strftime("%A")
            d.day, d.start, d.step = None, None, "date"
            return _say(L, "closed", weekday=wd)
        if d.start is not None:
            if not (hrs[0] <= d.start <= hrs[1] - config.MIN_STAY_MINUTES):
                hours = say_hours(d.day, L)
                d.start, d.step = None, "time"
                return _say(L, "outside", hours=hours)
            n = now()
            if d.day == n.date() and d.start < n.hour * 60 + n.minute + config.MIN_LEAD_MINUTES:
                d.start, d.step = None, "time"
                return _say(L, "past")
    return None


def _or_join(items: list[str], lang: str) -> str:
    word = " और " if lang == "hi" else " or "
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + word + items[-1]


def _availability_problem(d: Draft) -> str | None:
    """With kind, party, day and time known: is there room? If not, offer alternatives."""
    if None in (d.kind, d.party, d.day, d.start):
        return None
    L = d.lang
    if check_slot(d.day, d.start, d.party, d.kind, d.seat):
        return None
    if d.seat and (seat := get_seat(d.seat)):
        # the caller asked for this particular table and it is already booked at that time
        name, date_say, time_say = _seat_say(seat["name"], L), say_date(d.day, L), say_time(d.start, L)
        msg = _say(L, "seat_booked", seat=name, date=date_say, time=time_say)
        others = [_seat_say(n, L) for n in free_seat_names(d.day, d.start, d.party, d.kind)]
        if others:
            msg += _say(L, "seat_free", tables=_or_join(others, L), seat=name)
        else:
            alts = nearby_times(d.day, d.start, d.party, d.kind, seat_id=d.seat)
            if alts:
                msg += _say(L, "seat_times", seat=name, alts=_or_join([say_time(a, L) for a in alts], L))
        d.seat, d.start, d.step = None, None, "time"
        return msg
    alts = nearby_times(d.day, d.start, d.party, d.kind)
    kind_say, date_say = say_kind(d.kind, L), say_date(d.day, L)
    if alts:
        stale, d.start, d.step = d.start, None, "time"
        return _say(L, "alts", kind=kind_say, party=d.party, time=say_time(stale, L), date=date_say,
                    alts=_or_join([say_time(a, L) for a in alts], L))
    other = "special" if d.kind == "normal" else "normal"
    if nearby_times(d.day, d.start, d.party, other, 1):
        d.kind, d.step = None, "kind"
        return _say(L, "other", kind=kind_say, party=d.party, date=date_say, other=say_kind(other, L))
    d.day, d.start, d.step = None, None, "date"
    return _say(L, "full", date=date_say, party=d.party, people=_people(d.party, L))


def _extract(d: Draft, text: str) -> int:
    """Pull every answer we can from the sentence into the draft. Returns how many answers changed."""
    changed = 0
    t = text
    seat, t = parse_seat(t)
    if seat and seat["id"] != d.seat:
        d.seat, d.kind, changed = seat["id"], seat["kind"], changed + 1
    kind = parse_kind(t) if not seat else None
    if kind and kind != d.kind:
        d.kind, changed = kind, changed + 1
        if d.seat and (cur := get_seat(d.seat)) and cur["kind"] != kind:
            d.seat = None            # asked for the other table type: forget the specific table
    day, t = parse_date(t)
    if day and day != d.day:
        d.day, changed = day, changed + 1
    start, t = parse_time(t, bare_ok=(d.step == "time"), d=d.day)
    if start is not None and start != d.start:
        d.start, changed = start, changed + 1
    party = parse_party(t, bare_ok=(d.step == "party"))
    if party is not None and party != d.party:
        d.party, changed = party, changed + 1
    name = parse_name(text, bare_ok=(d.step == "name"))
    if name and name != d.name:
        d.name, changed = name, changed + 1
    return changed


def _chat_bits(session_id: str, d: Draft, text: str, via_voice: bool) -> Reply:
    L = d.lang
    t = text.strip()
    # a plain question ("what is special about the garden seats?") goes to the LLM, then we re-ask
    if d.step not in ("confirm", "offer") and _QUESTION_START.match(t) and not re.search(r"\d", t):
        return Reply(None, "cafe.booking_flow", suffix=_prompt(d))
    # ---- confirmation / offer steps
    if d.step == "offer":
        if _YES.search(t) and not _NO.search(t):
            d.step = "kind"
            _extract(d, t)
            problem = _validate(d) or _availability_problem(d)
            return Reply(problem or _prompt(d), "cafe.booking_flow")
        if _NO.search(t):
            _drafts.pop(session_id, None)
            return Reply(_say(L, "aborted"), "cafe.booking_flow")
    yes = bool(_YES.search(t)) and not _NO.search(t)
    no = bool(_NO.search(t)) and not _YES.search(t)
    changed = _extract(d, t)
    if changed:
        d.fresh = False
    problem = _validate(d)
    if problem is None:
        problem = _availability_problem(d)
    if problem:
        d.fails = 0
        return Reply(problem, "cafe.booking_flow")
    # ---- confirmed?
    if d.step == "confirm" and not changed:
        if yes:
            rec = create_booking(session_id, d.name, d.party, d.kind, d.day, d.start, d.seat)
            if rec is None:
                if d.seat:                 # the chosen table is gone: say which one and ask for another
                    return Reply(_availability_problem(d) or _prompt(d), "cafe.booking_flow")
                return Reply(_say(L, "taken") + (_availability_problem(d) or _prompt(d)), "cafe.booking_flow")
            _drafts.pop(session_id, None)
            names = [_seat_say(n, L) for n in rec["seat_names"]]
            seats = (" और " if L == "hi" else " and ").join(names)
            return Reply(_say(L, "booked", kind=say_kind(d.kind, L), party=d.party, people=_people(d.party, L),
                              date=say_date(d.day, L), time=say_time(d.start, L),
                              seats=seats, code=say_code(rec["code"])), "cafe.booking_flow", booking=rec)
        if no:
            return Reply(_say(L, "no_change"), "cafe.booking_flow")
    # ---- nothing understood?
    if not changed:
        if d.step != "confirm" and _QUESTION.search(t):
            return Reply(None, "cafe.booking_flow", suffix=_prompt(d))  # LLM answers, then we re-ask
        if d.fresh:                        # the booking was only just started: simply ask the first question
            d.fresh = False
            return Reply(_prompt(d), "cafe.booking_flow")
        d.fails += 1
        return Reply(_say(L, "retry") + _prompt(d), "cafe.booking_flow")
    d.fails = 0
    return Reply(_prompt(d), "cafe.booking_flow")


def handle(session_id: str, text: str, via_voice: bool = True) -> Reply | None:
    """Return the reply for a booking-related message, or None to let the normal LLM answer."""
    t = " ".join((text or "").split())
    if not t:
        return None
    d = _drafts.get(session_id)
    lang = detect_lang(t, d.lang if d else "en")
    in_flow = bool(d and d.active)

    if in_flow:
        d.lang = lang
        if config.VOICE_ONLY_BOOKING and not via_voice:
            return Reply(_say(lang, "voice_only"), "cafe.voice_only_notice")
        if d.step == "cancel_code":
            if re.search(r"(never ?mind|forget it|रहने दो|छोड़)", t, re.I):
                _drafts.pop(session_id, None)
                return Reply(_say(lang, "aborted"), "cafe.cancel_booking")
            return _cancel_by_code(session_id, d, t)
        if _ABORT.search(t):
            _drafts.pop(session_id, None)
            return Reply(_say(lang, "aborted"), "cafe.booking_flow")
        if _RESTART.search(t):
            d.kind = d.party = d.day = d.start = d.name = d.seat = None
            d.fails = 0
            return Reply(_say(lang, "restart") + _prompt(d), "cafe.booking_flow")
        if _HOURS.search(t) and not _BOOKVERB.search(t) and not parse_time(t, False, d.day)[0]:
            day = d.day or now().date()
            return Reply(_say(lang, "hours", hours=say_hours(day, lang), date=say_date(day, lang)) + " " + _prompt(d),
                         "cafe.hours")
        return _chat_bits(session_id, d, t, via_voice)

    wants_book = bool(_BOOKVERB.search(t) or _SEATWORD.search(t))
    # cancel an existing booking by its code
    if _CANCEL_WORD.search(t) and _BOOKING_NOUN.search(t):
        if config.VOICE_ONLY_BOOKING and not via_voice:
            return Reply(_say(lang, "voice_only"), "cafe.voice_only_notice")
        d = _drafts[session_id] = Draft(active=True, step="cancel_code", lang=lang)
        return _cancel_by_code(session_id, d, t, first=True)
    # "is a table free tomorrow at 7?"  -> answer from the database, offer to book
    if _AVAIL.search(t) and _SEATWORD.search(t) and not _BOOKVERB.search(t):
        return _availability_reply(session_id, t, lang)
    if wants_book:
        if config.VOICE_ONLY_BOOKING and not via_voice:
            return Reply(_say(lang, "voice_only"), "cafe.voice_only_notice")
        d = _drafts[session_id] = Draft(active=True, step="kind", lang=lang)
        return _chat_bits(session_id, d, t, via_voice)
    if _HOURS.search(t):
        day, _ = parse_date(t)
        day = day or now().date()
        return Reply(_say(lang, "hours", hours=say_hours(day, lang), date=say_date(day, lang)), "cafe.hours")
    return None


def _cancel_by_code(session_id: str, d: Draft, t: str, first: bool = False) -> Reply:
    code = parse_code(t)
    if not code:
        return Reply(_say(d.lang, "cancel_ask" if first else "cancel_bad"), "cafe.cancel_booking")
    rec = cancel_booking(code)
    if rec is None:
        return Reply(_say(d.lang, "cancel_bad"), "cafe.cancel_booking")
    _drafts.pop(session_id, None)
    day = date.fromisoformat(rec["date"])
    return Reply(_say(d.lang, "cancel_ok", code=say_code(code), name=rec["name"], date=say_date(day, d.lang),
                      time=say_time(rec["start_min"], d.lang)), "cafe.cancel_booking")


def _free_counts(day: date, start: int) -> dict[str, int]:
    hrs = hours_for(day)
    out = {}
    with _lock:
        conn = _connect()
        try:
            for kind in KINDS:
                out[kind] = len(_free_seats(conn, day, start, _end_for(start, hrs[1]), kind))
        finally:
            conn.close()
    return out


def _availability_reply(session_id: str, t: str, lang: str) -> Reply:
    day, rest = parse_date(t)
    start, rest = parse_time(rest, False, day)
    party = parse_party(rest, False)
    day = day or now().date()
    if start is None:
        return Reply(_say(lang, "avail_today", hours=say_hours(day, lang)), "cafe.availability")
    d = _drafts[session_id] = Draft(active=True, step="offer", lang=lang, day=day, start=start,
                                    party=party, kind=parse_kind(t))
    problem = _validate(d)
    if problem:
        _drafts.pop(session_id, None)
        return Reply(problem, "cafe.availability")
    counts = _free_counts(day, start)
    if counts["normal"] + counts["special"] == 0:
        d.start = None
        return Reply(_say(lang, "avail_no", date=say_date(day, lang), time=say_time(start, lang)), "cafe.availability")
    return Reply(_say(lang, "avail_yes", date=say_date(day, lang), time=say_time(start, lang),
                      n_normal=counts["normal"], n_special=counts["special"]), "cafe.availability")
