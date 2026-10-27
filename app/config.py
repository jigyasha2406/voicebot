"""Central configuration, read once from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- API keys -------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# --- Models ---------------------------------------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
# Optional ISO code ("hi", "en") to skip language auto-detection. Leave empty for Hindi/English mix.
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "")
# Short answers are the biggest speed win for a voice bot.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "90"))

# Hedging: if speech-to-text or the first voice chunk is slower than this, a duplicate request is
# started and the faster one wins. This is what trims the slow replies.
STT_HEDGE_SECONDS = float(os.getenv("STT_HEDGE_SECONDS", "0.6"))
TTS_HEDGE_SECONDS = float(os.getenv("TTS_HEDGE_SECONDS", "0.55"))

# --- Text to speech -------------------------------------------------------
# "elevenlabs" or "sarvam". The UI can switch per request if both keys exist.
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
# eleven_flash_v2_5 is ElevenLabs' low-latency model (~75 ms) and supports Hindi.
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
ELEVENLABS_FORMAT = os.getenv("ELEVENLABS_FORMAT", "mp3_22050_32")
SARVAM_SPEAKER = os.getenv("SARVAM_SPEAKER", "anushka")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "bulbul:v2")
SARVAM_SAMPLE_RATE = int(os.getenv("SARVAM_SAMPLE_RATE", "16000"))

# --- Storage --------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG_DIR = LOG_DIR / "chats"   # one .json and .txt per conversation
CHAT_LOG_DIR.mkdir(exist_ok=True)
DB_PATH = Path(os.getenv("METADATA_DB", LOG_DIR / "conversations.db"))

# --- Cafe ------------------------------------------------------------------
CAFE_NAME = os.getenv("CAFE_NAME", "Brew & Bloom Cafe")
CAFE_TZ = os.getenv("CAFE_TZ", "Asia/Kolkata")
CAFE_OPEN = os.getenv("CAFE_OPEN", "09:00")             # default opening hours, used only to seed the database
CAFE_CLOSE = os.getenv("CAFE_CLOSE", "22:00")           # (after that, edit them in the database / API)
BOOKING_MINUTES = int(os.getenv("BOOKING_MINUTES", "90"))     # how long a table is held
SLOT_STEP = int(os.getenv("SLOT_STEP", "30"))                 # alternative times are offered every N minutes
MIN_STAY_MINUTES = int(os.getenv("MIN_STAY_MINUTES", "45"))   # last booking starts this long before closing
MIN_LEAD_MINUTES = int(os.getenv("MIN_LEAD_MINUTES", "15"))   # same-day bookings need this much notice
MAX_PARTY = int(os.getenv("MAX_PARTY", "12"))
MAX_DAYS_AHEAD = int(os.getenv("MAX_DAYS_AHEAD", "30"))
# When true, a booking can only be made by speaking into the microphone (typed text gets a notice).
VOICE_ONLY_BOOKING = os.getenv("VOICE_ONLY_BOOKING", "true").lower() in ("1", "true", "yes")
# MySQL database for seats, opening hours and bookings (created automatically on first start)
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "cafe_voice_bot")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")   # if set, seat/hours edits need header X-Admin-Token

# Fallback only: the live prompt (with real opening hours and seat counts) comes from cafe.system_prompt().
SYSTEM_PROMPT = (
    "You are a helpful general-purpose voice assistant. "
    "Understand Hindi, English and Hinglish. Reply in the same language "
    "the user uses. Answer in one or two short sentences, spoken style, "
    "unless the user asks for detail: "
    "no markdown, no bullet points, no emojis. "
    "If you do not know something, say so honestly."
)

MAX_HISTORY_MESSAGES = 8  # less context = faster LLM first token


def available_tts_providers() -> list[str]:
    providers = []
    if ELEVENLABS_API_KEY:
        providers.append("elevenlabs")
    if SARVAM_API_KEY:
        providers.append("sarvam")
    providers.append("browser")  # free built-in browser voice, needs no key
    return providers


# Log of which function produced each answer (one JSON object per line).
TRACE_LOG = LOG_DIR / "answer_trace.jsonl"
