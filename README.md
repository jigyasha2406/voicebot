# Cafe seat-booking voice bot

This is the original voice bot (Groq speech-to-text + streaming LLM + ElevenLabs/Sarvam/browser voice, latency timer,
history, answer-trace log) turned into a **cafe seat-booking assistant**. Everything that was in the original still works;
the booking desk sits in front of the LLM. Anything that is not about booking (greetings, "what are your timings", general
questions) is still answered by the LLM, which now knows the cafe's real opening hours and seat counts.

## Booking by voice

Tap the microphone and say, in English, Hindi or Hinglish, for example:

> "I want to book a table" (then answer each question) or in one go: "Book a **special** table for **4** people **tomorrow** at **7:30 pm**"

The bot asks only for what is still missing, in this order:

1. **Special or normal table?** Special = window, garden, lounge or the private room (birthdays, celebrations). Normal = regular tables.
2. **How many people?**
3. **Which day?** today, tomorrow, a weekday, or a date
4. **What time?** (checked against that day's opening hours)
5. **Name for the booking**, then a spoken **confirmation**, then **"Done"** with the tables and a **4-digit booking code**.

Also by voice: "Is a table free tomorrow at 8 pm for 3 people?", "What are your timings?", "Cancel my booking" (then say the code),
"start over", "never mind". At the confirmation you can change anything ("make it 8 pm", "5 people") before saying yes.
During a booking the microphone **re-opens by itself** after each question and sends your answer when you pause
(untick *Keep listening during a booking* for tap-to-talk).

Booking is **voice only** by default: typed text can still ask questions and check availability, but gets a notice if it tries
to book. Set `VOICE_ONLY_BOOKING=false` in `.env` to allow typing (useful for testing without a microphone).

## The database (MySQL, created on first run)

1. Install and start MySQL (or MariaDB) and `pip install -r requirements.txt` (adds `PyMySQL`).
2. In `.env` set `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` (default `cafe_voice_bot`).
3. Start the app. It creates the database, the three tables and the sample seats/hours automatically. If MySQL is not reachable the app stops at start-up with a message saying so.

The conversation log and CSV (`logs/conversations.db`) stay in SQLite; only the cafe data lives in MySQL.

| Table | Holds |
|---|---|
| `seats` | `id, name, kind (normal/special), capacity, active`. 14 sample seats are created: 8 normal (2/4/6 seaters), 6 special (2 windows, 2 gardens, 1 lounge, 1 private room for 10). |
| `opening_hours` | one row per weekday (0 = Monday): `open_min, close_min, closed`. Seeded from `CAFE_OPEN` / `CAFE_CLOSE`. |
| `bookings` | `code, name, party, kind, date, start_min, end_min, seat_ids, status, session_id, created_at` |

**Asking for a particular table:** say e.g. "book Table 5 for 3 people tomorrow at 7 pm" or "Window 2" / "Garden 1" / "Lounge" / "Private room".
If that table is already booked at that time the bot says *"Sorry, Table 5 is already booked tomorrow at 7 PM. Please select a different table and time slot"*,
lists the tables that are free then, and asks again. The same message is spoken if the table is taken by someone else between
the confirmation and the final save, and (in Hindi) for Hindi/Hinglish callers. Without naming a table, a full slot gets the same
"already booked, select a different table or time slot" message plus the three nearest free times.

Availability is always computed from these tables: a seat is free if no confirmed booking overlaps the requested time
(a booking holds a table for `BOOKING_MINUTES`, default 90). The smallest table that fits is used; if no single table is big
enough, up to 3 free tables of the same type are joined. If the wanted time is taken, the bot offers the three nearest free
times, or the other table type. The final write re-checks availability inside a transaction, so two callers can never get the same table.
Bookings are kept separate from `logs/`, so "Delete all history" does not remove them.

### Managing seats and hours (REST, optional `X-Admin-Token`)

- `GET /api/cafe/overview?date=2026-10-25` seats with that day's bookings (feeds the page's side panel)
- `GET /api/cafe/availability?date=&time=19:30&party=4&kind=special`
- `GET /api/cafe/seats`, `POST /api/cafe/seats` `{"id":"N9","name":"Table 9","kind":"normal","capacity":4}`, `PATCH /api/cafe/seats/N9` `{"active":0}`
- `GET /api/cafe/hours`, `PUT /api/cafe/hours/6` `{"open":"10:00","close":"20:00","closed":false}` (6 = Sunday)
- `GET /api/cafe/bookings?date=`, `POST /api/cafe/bookings/{code}/cancel`

In the trace log and CSV, booking turns show `answer_function` as `cafe.booking_flow`, `cafe.availability`, `cafe.hours`,
`cafe.cancel_booking` or `cafe.voice_only_notice`. Booking turns skip the LLM, so they are also the fastest replies.
The latency clock starts when recording stops; in hands-free mode the ~1 s pause used to detect the end of your sentence is not counted.

---

# Original voice bot documentation

# Voice bot with frontend, metadata, latency timer and answer-trace log

FastAPI backend + browser frontend. Speech-to-text and the LLM run on Groq. The bot's voice comes from ElevenLabs or Sarvam AI (switch in the page).

## Run it

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env      # then add your own keys
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 and allow the microphone. Tap the mic, speak, tap again to send. (Microphone access works on `localhost` or HTTPS only.)

## What is saved for every turn

Stored in `logs/conversations.db` (SQLite) and downloadable as CSV from the page or at `/api/turns.csv`.

| Field | Meaning |
|---|---|
| `user_input` | What the user said (transcribed) or typed |
| `bot_response` | The bot's reply text |
| `date`, `time`, `timezone`, `timestamp_iso` | When the question reached the server |
| `stt_seconds` | Speech-to-text time (0 for typed input) |
| `llm_seconds` / `llm_first_sentence_seconds` | Full LLM stream time / time to the first sentence |
| `tts_seconds` / `tts_first_seconds` | Total voice work (chunks overlap) / voice time of the first sentence |
| `server_first_audio_seconds` | Request received until the first audio chunk is ready |
| `answer_function`, `call_chain` | Which function answered, and every function that ran |
| `server_total_seconds` | Whole server-side turn: STT + LLM + TTS + overhead |
| `client_time_to_audio_seconds` | **Total latency as the user feels it**: from the moment they stop speaking until the bot's voice starts playing. Measured in the browser, so it includes upload and download. |
| `playback_seconds` | How long the bot's audio played (not part of the latency) |
| `session_id` | The conversation this turn belongs to |
| `tts_provider`, `llm_model`, `status`, `error` | Which engines ran and whether it worked (`status` is `exit` when the user says "stop voice bot") |

The big number in the frontend is `client_time_to_audio_seconds`. The coloured bar splits it into speech-to-text, LLM, voice generation, and upload/download/browser time.

## Logs folder

`logs/` is created next to `app/` and filled automatically (see `logs/README.txt`). For every conversation, `logs/chats/` holds a readable `.txt` and a `.json` file with when the chat started, when it ended, the chat duration, every question and answer with date and time, the function that answered, the call chain, and all timings (speech-to-text, LLM, voice, server total, response time to first sound). `logs/chat_sessions.csv` has one summary row per chat. The files update after every reply, and deleting a chat in the page deletes its files too.

## History management

The left panel of the page lists every saved conversation, newest first.

- **Reopen and continue**: click a conversation. Its messages and timings reload, and the bot keeps the earlier context, even after the server restarts. The page also reopens your last conversation when you refresh.
- **Search**: matches conversation titles and anything said by you or the bot.
- **Rename**, **download CSV** (one conversation), **delete** (one conversation and its timing data).
- **New chat** starts a fresh conversation. **Delete all history** wipes everything after a confirmation.
- A conversation is titled from its first question, and appears in the list after that first reply.

## Which function answered? (answer-trace log)

Every turn appends one JSON line to `logs/answer_trace.jsonl` (also `GET /api/trace`, download at `/api/trace.log`; a summary line also goes to `logs/voicebot.log`). The same info is stored in the CSV columns `answer_function` and `call_chain`, and shown under each bot reply and in the table.

`answer_function` is the function that produced the reply text:

| Value | When |
|---|---|
| `providers.chat_stream` | Normal answer from the Groq LLM |
| `main.exit_handler` | User said "stop voice bot" (fixed goodbye) |
| `main.empty_transcription_fallback` | Nothing was heard (fixed "could not understand") |
| `main.llm_empty_reply_fallback` | LLM returned nothing |

`call_chain` lists every function that ran in order with seconds, e.g.
`main.chat_voice > main._turn_events > providers.transcribe(0.25s) > providers.chat_stream(0.36s) > providers.synthesize[elevenlabs]#0(0.20s)`.

## How the latency target (1 to 1.5 s to first sound) is reached

**Update:** after the short first chunk, later sentences are merged (about 110 characters per voice request) so a normal answer needs 2 voice calls, not 3-4. default LLM is now `llama-3.1-8b-instant`, `LLM_MAX_TOKENS=90`, and the first spoken chunk is cut at a comma after ~10 chars or a word boundary after ~18 chars. A free **Browser voice** option (Web Speech API) was added; it is used automatically when no ElevenLabs/Sarvam key is set, or if those fail.

The old flow waited for the whole answer before making any audio. Now:

1. **Streaming LLM** (`providers.chat_stream`): tokens arrive as they are generated.
2. **Sentence-by-sentence voice**: the first finished sentence (or first clause) goes to text-to-speech immediately; later sentences are voiced in parallel while the first one plays.
3. **Audio streamed to the browser** as NDJSON events; the browser plays chunks back to back.
4. **Short answers**: prompt asks for one or two sentences, `LLM_MAX_TOKENS=120`, history cut to 8 messages.
5. **Faster voice**: ElevenLabs `eleven_flash_v2_5`, `mp3_22050_32`, `optimize_streaming_latency=3`; Sarvam at 16 kHz.
6. **No cold connections**: one shared HTTP/2 client, warmed at start and re-pinged every 25 s.
7. **Smaller upload**: mono 16 kHz, 24 kbps recording.

Realistic budget: STT 0.25-0.4 s + first LLM sentence 0.3-0.5 s + first voice chunk 0.2-0.4 s + network 0.1 s = about 0.9-1.4 s. The page shows "to first sound" per reply, so you can check it on your own network. If it is still above 1.5 s, in order: set `GROQ_MODEL=llama-3.1-8b-instant`, set `STT_LANGUAGE=hi` (or `en`) if you use one language, and run the server in a region near the providers. Long answers do not slow the first sound, only the total.

## API

- `POST /api/chat/voice` — multipart `audio` (+ `session_id`, `tts_provider`). **Streams NDJSON events**: `transcript`, `text` (a sentence), `audio` (base64 chunk), then `done` (with full metadata) or `error`.
- `POST /api/chat/text` — `{"text": "...", "session_id": "...", "tts_provider": "..."}`, same event stream
- `POST /api/turns/{id}/client-metrics` — browser reports its measured latency
- `GET /api/turns`, `GET /api/turns.csv` (both accept `?session_id=`), `GET /api/stats`, `GET /api/config`
- `GET /api/trace`, `GET /api/trace.log`
- `GET /api/conversations` (`?q=` to search), `GET /api/conversations/{id}`, `PATCH /api/conversations/{id}` (`{"title": "..."}`), `DELETE /api/conversations/{id}`, `DELETE /api/conversations`

## Changes from the version you sent

- It was a local microphone script, not a FastAPI server. It is now a FastAPI app; the old script is kept in `legacy/`.
- Speech-to-text uses the Groq Whisper API instead of local `faster-whisper`, so the browser's audio works without a heavy local model.
- `requirements.txt` had `elevenlabs\pygame` on one line, which breaks installation. Replaced.
- Sarvam AI support is new. It follows Sarvam's documented text-to-speech API but has not been run against a live key.

## Keys

Never commit `.env`. If a key was ever shared in a chat or zip, rotate it in the Groq / ElevenLabs / Sarvam dashboard.
#   v o i c e b o t  
 