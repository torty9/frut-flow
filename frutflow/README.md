# früt Flow

A local, private voice-dictation tool for macOS. Hold a hotkey, speak, release,
and your words are typed into whatever app you're focused on.

Everything runs **on your machine** — Parakeet on the Apple-Silicon GPU by
default. No account, no subscription, no API keys, and no audio or text ever
leaves your computer. It's 100% free.

```
   hold ⌥ (right Option) ─► 🎤 record ─► 🧠 transcribe on-device (Parakeet, GPU)
        ─► ✨ cleanup ─► 🩹 fix your proper nouns
        ─► 📋 paste ─► 👀 watch your edits & learn them automatically
```

## Why it's accurate (the part that stops you editing)

Three layers stack so you rarely have to fix anything — all **on-device, automatic**:

1. **A strong, fast model by default.** NVIDIA **Parakeet** (`parakeet-tdt-0.6b`)
   running on the Apple-Silicon **GPU** via MLX — fast *and* accurate, with
   punctuation and capitalization built in. It renders names as near-misses
   (e.g. *Vercel* → "Versal") rather than unrelated words, which the next layer
   repairs. (`faster-whisper distil-large-v3` is still available as a CPU fallback.)
2. **Phonetic proper-noun repair.** After transcription, near-miss words are
   snapped to your known vocabulary using sound-alike + edit-distance matching
   ("Versal" → "Vercel", "Frut" → "früt"), with guards so real words like
   *versatile* or *Boston* are never touched.
3. **Automatic learning from your edits.** When you fix a word right after it's
   pasted, früt Flow notices, confirms it's a genuine mis-hear (not a change of
   mind) by how similar the two words *sound*, and learns it — so it never gets
   that word wrong again. **You never run a "teach" command.**

## Requirements

- macOS 11+ on Apple Silicon
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
**first run downloads the speech model once** (~600 MB for the default Parakeet
model); after that it works fully offline.

You can also just double-click **`Start früt Flow.command`** in Finder to launch
it in a Terminal window.

### Manual install (if you prefer)

```bash
brew install portaudio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python flow.py --setup
python flow.py
```

## Grant macOS permissions (one-time)

In **System Settings ▸ Privacy & Security**, grant these to the program that
launches `flow.py` (Terminal, iTerm, or VS Code):

- **Microphone** — to hear you
- **Accessibility** — to paste/type into other apps
- **Input Monitoring** — to detect the global hotkey

After granting, **fully quit and reopen** that program.

## Usage

1. Run `./run.sh`. You'll see `früt Flow is running.`
2. Click into any text field (Slack, browser, Notes, your editor…).
3. **Hold Right Option, speak, then release.** A second later the cleaned-up
   text is pasted at your cursor.

Quit with `Ctrl-C`.

## Configuration

Edit `~/.flowdictate/config.json` (see `config.example.json`, or use the in-app
**Settings** window). The most-used keys:

| Key | Default | Notes |
|---|---|---|
| `hotkey` | `"alt_r"` | Any pynput key name (`alt_r`, `cmd_r`, `ctrl_r`, `f6`) or single char. Use a **modifier** so holding it doesn't type. |
| `mode` | `"hold"` | `"hold"` = push-to-talk · `"toggle"` = tap to start/stop |
| `transcribe_backend` | `"parakeet"` | `"parakeet"` (NVIDIA Parakeet on the GPU — fastest, default) · `"local"` (faster-whisper on CPU). Both on-device. |
| `parakeet_model` | `"mlx-community/parakeet-tdt-0.6b-v2"` | v2 = English (best English accuracy) · `...-v3` = 25 languages |
| `model` | `"distil-large-v3"` | faster-whisper model, used when `transcribe_backend: "local"`. `medium.en`/`small.en` faster, miss more names. |
| `normalize_method` | `"rms"` | `"rms"` = average-loudness normalize + soft-limit (best for quiet/whispered) · `"peak"` = old peak-normalize |
| `fuzzy_correct` | `true` | Phonetic proper-noun repair against your learned vocab (engine-agnostic) |
| `learn_from_edits` | `true` | Auto-learn corrections by watching the field you paste into |
| `cleanup` | `"basic"` | `"none"` (raw) · `"basic"` (strip fillers) · `"local"` (on-device misheard-word repair — no cloud, no key) |
| `insert_method` | `"paste"` | `"paste"` (clipboard+Cmd-V) or `"type"` (key-by-key) |
| `restore_clipboard` | `true` | Put your previous clipboard back after pasting the dictation |
| `auto_space` | `true` | Prepend a space so dictation merges naturally with existing text |
| `undo_phrases` | `["never mind", …]` | Whole-utterance phrases that delete the previous dictation instead of typing |
| `show_hud` | `true` | Floating waveform pill near the bottom of the screen while recording |
| `appearance` | `"dark"` | UI theme for the app's own windows (`"dark"`, `"light"`, `"auto"`) |
| `max_record_seconds` | `120` | Safety cap that auto-stops a runaway capture (a warning sound plays ~10 s before) |

### Accuracy vs. speed

The default **Parakeet** backend runs on the Apple-Silicon **GPU** and transcribes
a typical clip in well under a second, versus ~2–5 s for the CPU-bound
`faster-whisper` path — while being at least as accurate. The model loads (and
warms up) once at startup.

Not sure it's better on *your* voice — especially if you speak quietly? Compare
them on the same clip:

```bash
./run.sh --compare        # records ~6 s, transcribes with BOTH engines, prints both
```

To use `faster-whisper` instead, set `"transcribe_backend": "local"`. The fuzzy
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

### Optional on-device cleanup (still 100% local, still free)

There are no cloud back-ends and no API keys anywhere in this app. The one
optional extra is a **fully on-device** cleanup step:

- **On-device misheard-word repair** — a small local model reads the whole
  dictation and fixes clear mishearings (homophones like *there/their*,
  *pier/peer*, or a garbled term the sentence makes obvious). It's conservative:
  it never paraphrases, answers, or touches a word that was already right. No
  cloud, no account, no API key:

  ```bash
  pip install "mlx-lm>=0.28" "transformers>=4.44,<5"
  ```
  then set `"cleanup": "local"`. The repair model (~0.9 GB) downloads once on
  first use, then runs entirely on your Apple-Silicon GPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Hotkey does nothing | Grant **Input Monitoring** to your terminal, then restart it. |
| Nothing gets pasted | Grant **Accessibility**. Try `"insert_method": "type"`. |
| `PortAudioError` / no audio | `brew install portaudio`; check `python flow.py --list-devices`. |
| First run is slow | It's downloading the speech model once; subsequent runs are instant. |
| Dictation feels slow | The default Parakeet backend is already sub-second/clip. If you switched to `"local"` (faster-whisper), that's the ~2–5 s CPU path — switch back to `"parakeet"`. |
| It keeps misspelling a name | Just fix it once after it pastes — it learns the correction automatically. Or `./run.sh --correct "heard" "correct"`. |
| A real word gets "corrected" | Raise `"fuzzy_threshold"` (e.g. `0.85`), or set `"fuzzy_correct": false`. |
| Old clipboard isn't restored | `restore_clipboard` is on by default; if it misses on a slow Mac, set `"restore_clipboard": false` for dictation-only behavior. |
| Transcription too aggressive | Set `"cleanup": "none"` to get verbatim model output. |

## Privacy

**Nothing ever leaves your computer** — with any setting. There are no cloud
back-ends and no API keys anywhere in this app; all transcription and cleanup
runs on-device. After the one-time model download there are no network calls at
all. Your learned vocabulary, corrections, history, and log live in
`~/.flowdictate` and are written owner-only (`0600`). That's the main reason this
exists.

## Credits & licenses

- **früt Flow** is released under the [MIT License](../LICENSE).
- **Speech models:** [NVIDIA Parakeet](https://huggingface.co/nvidia) via
  [Apple MLX](https://github.com/ml-explore/mlx) /
  [parakeet-mlx](https://github.com/senstella/parakeet-mlx); optional
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) fallback.
- **Icons:** [Phosphor Icons](https://phosphoricons.com) — MIT License,
  © Phosphor Icons.
- **Libraries:** rapidfuzz, jellyfish, sounddevice, pynput, numpy, pyobjc.
