import io
import json
import logging
import os
import time
import uuid
import csv
from datetime import datetime, timezone
from pathlib import Path

import pygame
import sounddevice as sd
import scipy.io.wavfile as wavfile
from dotenv import load_dotenv
from groq import Groq
from elevenlabs.client import ElevenLabs
from faster_whisper import WhisperModel

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "8"))
AUDIO_DIR = BASE_DIR / "audio"
LOG_DIR = BASE_DIR / "logs"
METADATA_JSONL = LOG_DIR / "conversation_metadata.jsonl"
METADATA_CSV = LOG_DIR / "conversation_metadata.csv"
AUDIO_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "voicebot.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY is missing. Copy .env.example to .env and add your key.")

groq_client = Groq(api_key=GROQ_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

print("Loading Whisper speech model...")
speech_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded.")

conversation = [{
    "role": "system",
    "content": (
        "You are a helpful general-purpose voice assistant. "
        "Understand Hindi, English and Hinglish. Reply in the same language "
        "the user uses. Keep answers clear, natural and concise for speech. "
        "If you do not know something, say so honestly."
    )
}]

CSV_FIELDS = [
    "turn_id", "timestamp_start_local", "timestamp_end_local",
    "user_input", "bot_response", "date", "time", "timezone",
    "transcription_seconds", "llm_seconds", "tts_generation_seconds",
    "playback_seconds", "total_seconds", "status", "error"
]

def now_local_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def persist_metadata(record):
    """Append one metadata record to JSONL and CSV; never log API keys."""
    try:
        with METADATA_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_header = not METADATA_CSV.exists() or METADATA_CSV.stat().st_size == 0
        with METADATA_CSV.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({key: record.get(key, "") for key in CSV_FIELDS})
    except Exception:
        logging.exception("Could not persist conversation metadata")

def record_audio():
    input("\nPress ENTER to start recording...")
    print(f"Listening for {RECORD_SECONDS} seconds...")
    recording = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    audio_path = AUDIO_DIR / "input.wav"
    wavfile.write(str(audio_path), SAMPLE_RATE, recording)
    return str(audio_path)

def transcribe_audio(audio_path):
    started = time.perf_counter()
    segments, info = speech_model.transcribe(
        audio_path, beam_size=3, vad_filter=True
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed = time.perf_counter() - started
    print("Detected language:", info.language)
    print("You:", text)
    return text, elapsed

def get_ai_response(user_text):
    conversation.append({"role": "user", "content": user_text})
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=conversation,
            temperature=0.5,
            max_tokens=500,
        )
        answer = response.choices[0].message.content or "Sorry, I could not generate an answer."
        answer = answer.strip()
        conversation.append({"role": "assistant", "content": answer})
        if len(conversation) > 13:
            conversation[:] = [conversation[0]] + conversation[-12:]
        return answer
    except Exception:
        logging.exception("Groq API request failed")
        conversation.pop()
        raise

def generate_and_play_speech(text):
    """Return separate TTS-generation and audio-playback durations."""
    tts_start = time.perf_counter()
    audio_chunks = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=TTS_MODEL,
        output_format="mp3_44100_128",
    )
    audio_bytes = b"".join(audio_chunks)
    tts_seconds = time.perf_counter() - tts_start

    pygame.mixer.music.load(io.BytesIO(audio_bytes), "mp3")
    playback_start = time.perf_counter()
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    playback_seconds = time.perf_counter() - playback_start
    return tts_seconds, playback_seconds

def speak_startup(text):
    print("\nBot:", text)
    try:
        generate_and_play_speech(text)
    except Exception:
        logging.exception("Startup speech failed")
        print("Could not generate startup speech. Check ElevenLabs key/credits.")

def main():
    pygame.mixer.init()
    print("\n===================================")
    print("    HINDI + ENGLISH AI VOICE BOT")
    print("===================================")
    print(f"LLM: Groq ({GROQ_MODEL})")
    print(f"Voice: ElevenLabs ({TTS_MODEL})")
    print(f"Metadata JSONL: {METADATA_JSONL}")
    print(f"Metadata CSV:   {METADATA_CSV}")
    print("Say 'stop voice bot' to exit.")
    speak_startup("Namaste! Your AI voice assistant is ready.")

    try:
        while True:
            turn_id = str(uuid.uuid4())
            turn_start_wall = now_local_iso()
            turn_start_perf = time.perf_counter()
            date_now = datetime.now().astimezone()
            record = {
                "turn_id": turn_id,
                "timestamp_start_local": turn_start_wall,
                "timestamp_end_local": "",
                "user_input": "",
                "bot_response": "",
                "date": date_now.date().isoformat(),
                "time": date_now.strftime("%H:%M:%S %Z"),
                "timezone": str(date_now.tzinfo),
                "transcription_seconds": 0.0,
                "llm_seconds": 0.0,
                "tts_generation_seconds": 0.0,
                "playback_seconds": 0.0,
                "total_seconds": 0.0,
                "status": "started",
                "error": "",
            }
            try:
                audio_path = record_audio()
                user_text, stt_seconds = transcribe_audio(audio_path)
                record["transcription_seconds"] = round(stt_seconds, 4)
                record["user_input"] = user_text

                if not user_text:
                    answer = "Sorry, I could not understand. Please try again."
                    record["bot_response"] = answer
                    print("\nBot:", answer)
                    try:
                        tts, playback = generate_and_play_speech(answer)
                        record["tts_generation_seconds"] = round(tts, 4)
                        record["playback_seconds"] = round(playback, 4)
                    except Exception as exc:
                        record["error"] = f"TTS error: {type(exc).__name__}: {exc}"
                        logging.exception("TTS failed for empty transcription")
                    record["status"] = "empty_transcription"
                else:
                    command = user_text.lower().strip()
                    exit_commands = ["stop voice bot", "exit voice bot", "बंद करो", "बंद कर दो", "अलविदा"]
                    if any(word in command for word in exit_commands):
                        answer = "Goodbye! फिर मिलेंगे।"
                        record["bot_response"] = answer
                        print("\nBot:", answer)
                        try:
                            tts, playback = generate_and_play_speech(answer)
                            record["tts_generation_seconds"] = round(tts, 4)
                            record["playback_seconds"] = round(playback, 4)
                        except Exception as exc:
                            record["error"] = f"TTS error: {type(exc).__name__}: {exc}"
                            logging.exception("TTS failed on exit")
                        record["status"] = "exit"
                        record["timestamp_end_local"] = now_local_iso()
                        record["total_seconds"] = round(time.perf_counter() - turn_start_perf, 4)
                        persist_metadata(record)
                        break

                    llm_start = time.perf_counter()
                    answer = get_ai_response(user_text)
                    record["llm_seconds"] = round(time.perf_counter() - llm_start, 4)
                    record["bot_response"] = answer
                    print("\nBot:", answer)

                    try:
                        tts, playback = generate_and_play_speech(answer)
                        record["tts_generation_seconds"] = round(tts, 4)
                        record["playback_seconds"] = round(playback, 4)
                        record["status"] = "success"
                    except Exception as exc:
                        record["status"] = "tts_error"
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        logging.exception("ElevenLabs speech error")
                        print("Could not generate/play speech. Check ElevenLabs API key and credits.")

                record["timestamp_end_local"] = now_local_iso()
                # End-to-end time from start of the turn (before recording) until audio playback finishes.
                record["total_seconds"] = round(time.perf_counter() - turn_start_perf, 4)
                persist_metadata(record)
                print(
                    "\nLatency breakdown (seconds): "
                    f"transcription={record['transcription_seconds']:.2f}, "
                    f"LLM={record['llm_seconds']:.2f}, "
                    f"TTS generation={record['tts_generation_seconds']:.2f}, "
                    f"audio playback={record['playback_seconds']:.2f}, "
                    f"TOTAL={record['total_seconds']:.2f}"
                )
                logging.info(
                    "turn_id=%s status=%s total_seconds=%.4f stt=%.4f llm=%.4f tts=%.4f playback=%.4f",
                    turn_id, record["status"], record["total_seconds"],
                    record["transcription_seconds"], record["llm_seconds"],
                    record["tts_generation_seconds"], record["playback_seconds"]
                )
            except KeyboardInterrupt:
                print("\nVoice bot stopped.")
                break
            except Exception as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["timestamp_end_local"] = now_local_iso()
                record["total_seconds"] = round(time.perf_counter() - turn_start_perf, 4)
                persist_metadata(record)
                logging.exception("Unexpected turn error")
                print("An error occurred. Check logs/voicebot.log")
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    main()
