# Wispr DIY

A local, private voice-dictation tool for macOS — a from-scratch replacement
for **Wispr Flow**. Hold a hotkey, speak, release, and your words are typed into
whatever app you're focused on.

Unlike Wispr, transcription runs **on your machine** by default
(`faster-whisper`). No account, no subscription, and no audio leaves your
computer unless you explicitly switch to a cloud back-end.

```
   hold ⌥ (right Option) ─► 🎤 record ─► 🧠 transcribe on-device (Parakeet, GPU)
        ─► ✨ cleanup ─► 🩹 fix your proper nouns
        ─► 📋 paste ─► 👀 watch your edits & learn them automatically
```

## How it maps to Wispr Flow

| Wispr Flow | Wispr DIY |
|---|---|
| Push-to-talk hotkey (Fn) | Push-to-talk hotkey (Right Option by default, configurable) |
| Cloud ASR (always online) | **On-device** NVIDIA `Parakeet` on the Apple-Silicon GPU (offline) — or `faster-whisper`, or optional OpenAI Whisper |
| Llama-based AI cleanup | Local cleanup + phonetic proper-noun repair — or optional Anthropic LLM cleanup |
| Personal Dictionary with auto-add | **Automatic** learning from your edits — no manual teaching (see below) |
| Context-aware formatting | Active-app context fed to the optional LLM cleanup |
| Inserts via Accessibility | Inserts via clipboard + Cmd-V (Accessibility permission) |
| $15/mo Pro | Free |

## Why it's accurate (the part that stops you editing)

Three layers stack so you rarely have to fix anything — all **on-device, automatic**:

1. **A strong, fast model by default.** NVIDIA **Parakeet** (`parakeet-tdt-0.6b`)
   running on the Apple-Silicon **GPU** via MLX — ~6–12× faster than the old
   `faster-whisper` CPU path *and* more accurate, with punctuation and
   capitalization built in. Like the old default it renders names as near-misses
   (e.g. *Vercel* → "Versal") rather than unrelated words, which the next layer
   repairs. (`faster-whisper distil-large-v3` is still available as a fallback.)
2. **Phonetic proper-noun repair.** After transcription, near-miss words are
   snapped to your known vocabulary using sound-alike + edit-distance matching
   ("Versal" → "Vercel", "Frut" → "früt"), with guards so real words like
   *versatile* or *Boston* are never touched.
3. **Automatic learning from your edits.** When you fix a word right after it's
   pasted, Wispr DIY notices, confirms it's a genuine mis-hear (not a change of
   mind) by how similar the two words *sound*, and learns it — so it never gets
   that word wrong again. **You never run a "teach" command.**

## Requirements

- macOS 11+
- [Homebrew](https://brew.sh) (used to install the PortAudio audio library)
- Python 3.9+

## Quick start

```bash
cd "frutflow"
chmod +x run.sh
./run.sh --setup     # writes ~/.flowdictate/config.json + prints permissions
./run.sh             # creates a venv, installs deps, and runs
```

`run.sh` makes a `.venv`, installs the Python deps, and launches the app. The
**first run downloads the Whisper model** (~150 MB for `base.en`); after that it
works fully offline.

### Manual install (if you prefer)

```bash
brew install portaudio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python flow.py --setup
python flow.py
```

## Grant macOS permissions (one-time)

This is the same permission model Wispr uses. In
**System Settings ▸ Privacy & Security**, grant these to the program that
launches `flow.py` (Terminal, iTerm, or VS Code):

- **Microphone** — to hear you
- **Accessibility** — to paste/type into other apps
- **Input Monitoring** — to detect the global hotkey

After granting, **fully quit and reopen** that program.

## Usage

1. Run `./run.sh`. You'll see `Wispr DIY is running.`
2. Click into any text field (Slack, browser, Notes, your editor…).
3. **Hold Right Option, speak, then release.** A second later the cleaned-up
   text is pasted at your cursor.

Quit with `Ctrl-C`.

## Configuration

Edit `~/.flowdictate/config.json` (see `config.example.json`):

| Key | Default | Notes |
|---|---|---|
| `hotkey` | `"alt_r"` | Any pynput key name (`alt_r`, `cmd_r`, `ctrl_r`, `f6`) or single char. Use a **modifier** so holding it doesn't type. |
| `mode` | `"hold"` | `"hold"` = push-to-talk · `"toggle"` = tap to start/stop |
| `transcribe_backend` | `"parakeet"` | `"parakeet"` (NVIDIA Parakeet on the GPU — fastest, default) · `"local"` (faster-whisper on CPU) · `"openai"` (cloud) |
| `parakeet_model` | `"mlx-community/parakeet-tdt-0.6b-v2"` | v2 = English (best English accuracy) · `...-v3` = 25 languages |
| `model` | `"distil-large-v3"` | faster-whisper model, used when `transcribe_backend: "local"`. `medium.en`/`small.en` faster, miss more names. |
| `normalize_method` | `"rms"` | `"rms"` = average-loudness normalize + soft-limit (best for quiet/whispered) · `"peak"` = old peak-normalize |
| `vocab_biasing` | `"hotwords"` | faster-whisper only (Parakeet has no decoder biasing; the fuzzy layer covers it) |
| `fuzzy_correct` | `true` | Phonetic proper-noun repair against your learned vocab (engine-agnostic) |
| `learn_from_edits` | `true` | Auto-learn corrections by watching the field you paste into |
| `cleanup` | `"basic"` | `"none"` (raw) · `"basic"` (strip fillers) · `"llm"` (Anthropic polish) |
| `insert_method` | `"paste"` | `"paste"` (clipboard+Cmd-V) or `"type"` (key-by-key) |
| `max_record_seconds` | `120` | Safety cap that auto-stops a runaway capture (a warning sound plays ~10 s before) |

### Accuracy vs. speed

The default **Parakeet** backend runs on the Apple-Silicon **GPU** and transcribes
a typical clip in well under a second (~0.15–1.3 s), versus ~2–5 s for the old
CPU-bound `faster-whisper` path — while being at least as accurate. The model
loads (and warms up) once at startup.

Not sure it's better on *your* voice — especially if you speak quietly? Compare
them on the same clip:

```bash
./run.sh --compare        # records ~6 s, transcribes with BOTH engines, prints both
```

To go back to `faster-whisper`, set `"transcribe_backend": "local"`. The fuzzy
corrector and auto-learning work identically with either engine.

### Transcribe an existing audio file

No mic needed — point it at a recording (voice memo, meeting clip, anything macOS
can read: wav, aiff, m4a, mp3, …):

```bash
./run.sh --transcribe memo.m4a          # prints the transcript
./run.sh --transcribe memo.m4a --copy   # …and copies it to the clipboard
./run.sh --transcribe memo.m4a 2>/dev/null > out.txt   # transcript only, for scripting
```

It runs the same engine + correction pipeline as live dictation; status goes to
stderr so stdout is just the text.

### See / shape what it has learned

```bash
./run.sh --show-learning            # your learned vocab, corrections, hotwords
./run.sh --try "some Versal text"   # dry-run the correction pipeline on a string
```

You usually won't need to teach it anything — it learns from your edits. But you
still can: `./run.sh --correct "heard" "correct"` (or double-click
**Teach a Word.command**).

### Optional cloud back-ends

- **OpenAI Whisper** (better accuracy, sends audio to OpenAI):
  `pip install openai`, `export OPENAI_API_KEY=...`, set
  `"transcribe_backend": "openai"`.
- **Anthropic cleanup** (Wispr-style formatting, sends text to Anthropic):
  `pip install anthropic`, `export ANTHROPIC_API_KEY=...`, set `"cleanup":
  "llm"`. The prompt is tuned to **only** fix punctuation/casing/fillers and to
  honor your known spellings — it is explicitly forbidden from rewriting your
  words (the "the AI changed what I said" failure mode). It also gets the active
  app name for tone. This is the one feature that sends text off-device, so it's
  off by default.

## Run it in the background / on login

Once you're happy with it, you can launch it at login with a small
`launchd` agent, or just keep a Terminal tab open running `./run.sh`. (Ask and
I'll generate a `LaunchAgent` plist for you.)

## Troubleshooting

| Symptom | Fix |
|---|---|
| Hotkey does nothing | Grant **Input Monitoring** to your terminal, then restart it. |
| Nothing gets pasted | Grant **Accessibility**. Try `"insert_method": "type"`. |
| `PortAudioError` / no audio | `brew install portaudio`; check `python flow.py --list-devices`. |
| First run is slow | It's downloading the Whisper model once; subsequent runs are instant. |
| Dictation feels slow | The default Parakeet backend is already ~0.15–1.3 s/clip. If you switched to `"local"` (faster-whisper), that's the ~2–5 s CPU path — switch back to `"parakeet"`. |
| It keeps misspelling a name | Just fix it once after it pastes — it learns the correction automatically. Or `./run.sh --correct "heard" "correct"`. |
| A real word gets "corrected" | Raise `"fuzzy_threshold"` (e.g. `0.85`), or set `"fuzzy_correct": false`. |
| Pastes over my clipboard | `restore_clipboard` is on by default; increase the restore delay if your Mac is slow. |
| Transcription too aggressive | Set `"cleanup": "none"` to get verbatim model output. |

## Privacy

With the defaults (`transcribe_backend: local`, `cleanup: basic`) **nothing
leaves your computer** — no network calls at all after the one-time model
download. That's the main reason this exists.
