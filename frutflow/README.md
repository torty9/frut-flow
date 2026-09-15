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

- macOS 14 (Sonoma)+ on Apple Silicon
- [Homebrew](https://brew.sh) (used to install the PortAudio audio library)
- Python 3.10+

## Easiest install for friends

Build the unsigned installer disk image:

```bash
./package-dmg.sh
```

Send the generated `dist/frut-flow-mac-installer-YYYY-MM-DD.dmg`. Your friend
opens the disk image and double-clicks **`Install frut Flow.command`**. It
offers to install Homebrew and Python when needed, installs the local
dependencies, creates `~/Applications/frutflow.app`, enables on-device repair
cleanup, and downloads the local models once:

- Parakeet for speech recognition
- Qwen for on-device correction/cleanup

No API key or cloud code is needed. Because the disk image is not Developer ID
signed or notarized, macOS may require the one-time **System Settings ▸ Privacy
& Security ▸ Open Anyway** approval.

For the smallest possible attachment, send **`Install frut Flow from
GitHub.command`** instead. It downloads the latest app from the GitHub `main`
branch, so commit and push the current repo before sending it.

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

## Make a shareable installer zip

From this project folder:

```bash
./package-mac.sh
```

That creates a clean download at `dist/frut-flow-mac-share-YYYY-MM-DD.zip`.
Send or upload that zip. The recipient unzips it, double-clicks
**`Install frut Flow.command`**, and the installer copies the app files to
`~/Applications/frut-flow`, creates `~/Applications/frutflow.app`, installs the
Python dependencies, enables on-device repair cleanup, preloads Parakeet and
Qwen, and shows the one-time macOS permission steps.

Because this is not Developer ID signed/notarized, macOS may show an "Apple
cannot verify" warning. For a no-warning public download, build a signed and
notarized `.dmg` or `.pkg` with an Apple Developer account.

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

## Languages (English, Spanish, and 23 more)

Pick your language in **Settings ▸ Model ▸ Spoken language**:

- **Auto** — the engine detects the language of each dictation. Great if you
  switch between English and Spanish mid-day. Selecting it also switches the
  Parakeet model to the **Multilingual** v3 checkpoint (25 languages, Spanish
  included), which detects the spoken language on its own and punctuates
  Spanish properly (`¿…?`, `¡…!`, accents).
- **English / Español** — lock it to one language (slightly more accurate if
  you only ever use one).

Details the app handles for you:

- Choosing Auto or Español automatically uses a **multilingual model**: the
  English-only Parakeet v2 is swapped for v3, and on the Whisper backend the
  English-only `distil-large-v3` is swapped for `large-v3-turbo`. Each swap is
  a one-time model download; restart the app after changing the language.
- Cleanup is language-aware: Spanish text gets Spanish spoken-punctuation
  commands, RAE-style spacing for `¿ ¡`, and a fail-closed Spanish filler list
  (words like *este*, *pues*, or *eh* are never stripped).
- The classic Whisper silence hallucinations, English **and** Spanish
  ("Thanks for watching!", "Subtítulos realizados por la comunidad de
  Amara.org", "[Música]"…), are filtered out in every cleanup mode.
- Voice undo also understands Spanish: say **"borra eso"**, **"olvídalo"**,
  **"elimina eso"**, or **"deshaz eso"**.

## Spoken punctuation

The engines already punctuate from your pauses and intonation. On top of that,
`cleanup: "basic"` (the default) converts punctuation you *say*, like Apple
dictation or Wispr Flow:

| You say (English) | You get |
|---|---|
| "quote … end quote" (or "unquote" / "close quote") | `"…"` |
| "open quote" with no closer | quotes the rest of the utterance |
| "the quote unquote expert" | the "expert" |
| "period" · "full stop" · "comma" · "colon" · "semicolon" | `.` `,` `:` `;` |
| "question mark" · "exclamation point" | `?` `!` |
| "new line" · "new paragraph" | line / paragraph break |
| "dot dot dot" / "ellipsis" · "dash" · "hyphen" | `…` ` - ` `-` |
| "john at sign gmail dot com" | `john@gmail.com` |
| "open paren … close paren" · "underscore" · "hashtag" | `(…)` `_` `#` |

| Dices (español) | Sale |
|---|---|
| "abrir comillas … cerrar comillas" (o "comillas de apertura/cierre") | `"…"` |
| "punto" · "coma" · "dos puntos" · "punto y coma" | `.` `,` `:` `;` |
| "signo de interrogación" · "signo de exclamación/admiración" | `?` `!` |
| "abrir interrogación" · "abrir exclamación" | `¿` `¡` |
| "punto y aparte" | `.` + new paragraph |
| "punto y seguido" · "puntos suspensivos" | `.` `…` |
| "nueva línea" · "nuevo párrafo" | salto de línea / párrafo |
| "maría arroba gmail punto com" | `maría@gmail.com` |
| "guion bajo" / "barra baja" · "almohadilla" | `_` `#` |

Safety first: ambiguous words only convert when the context says *command*, not
prose — "the **trial period** ends", "**punto de vista**", "colon **cancer**",
"en **coma**", "ganamos por **dos puntos**" are all left exactly as spoken. A
bare "quote"/"comillas" needs a matching closer ("end quote"/"cerrar comillas")
so "get a quote from the plumber" is never touched.

## Configuration

Edit `~/.flowdictate/config.json` (see `config.example.json`, or use the in-app
**Settings** window). The most-used keys:

| Key | Default | Notes |
|---|---|---|
| `hotkey` | `"alt_r"` | Use Settings ▸ Dictation ▸ **Change**, then press the single key you want. Regular keys are stored as `vk:N`; modifier names like `alt`, `cmd`, `ctrl`, `shift`, or `cmd_r` also work in JSON. |
| `mode` | `"hold"` | `"hold"` = push-to-talk · `"toggle"` = tap to start/stop |
| `transcribe_backend` | `"parakeet"` | `"parakeet"` (NVIDIA Parakeet on the GPU — fastest, default) · `"local"` (faster-whisper on CPU). Both on-device. |
| `language` | `"en"` | `"en"` · `"es"` · any ISO code · `"auto"` (detect per dictation). Non-English auto-selects a multilingual model. |
| `parakeet_model` | `"mlx-community/parakeet-tdt-0.6b-v2"` | v2 = English (best English accuracy) · `...-v3` = 25 languages incl. Spanish (auto-detects) |
| `model` | `"distil-large-v3"` | faster-whisper model, used when `transcribe_backend: "local"`. English-only; swapped for `large-v3-turbo` automatically when `language` isn't `"en"`. |
| `normalize_method` | `"rms"` | `"rms"` = average-loudness normalize + soft-limit (best for quiet/whispered) · `"peak"` = old peak-normalize |
| `fuzzy_correct` | `true` | Conservative English name repair; skips real words, technical literals, and ambiguous matches. Explicit teachings work in every language. |
| `learn_from_edits` | `true` | Learn name fixes after two matching edits in separate dictations, limited to the inserted text |
| `cleanup` | `"basic"` | `"none"` (raw) · `"basic"` (fillers, spacing, caps + spoken punctuation like "quote … end quote") · `"local"` (on-device misheard-word repair — no cloud, no key) |
| `insert_method` | `"paste"` | `"paste"` (clipboard+Cmd-V), `"type"` (key-by-key), or `"clipboard"` (copy only — you press Cmd-V; needs **no** Accessibility permission) |
| `restore_clipboard` | `true` | Put your previous clipboard back after pasting the dictation |
| `auto_space` | `true` | Prepend a space so dictation merges naturally with existing text |
| `undo_phrases` | `["never mind", …]` | Whole-utterance phrases that delete the previous dictation instead of typing |
| `history_enabled` | `true` | Store the last 10 dictations locally for the History window. Set `false` for private mode. |
| `show_hud` | `true` | Floating waveform pill near the bottom of the screen while recording |
| `appearance` | `"dark"` | UI theme for the app's own windows (`"dark"`, `"light"`, `"system"`) |
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

Name fixes are learned after the same edit in two separate dictations. To teach
an authoritative replacement immediately, use: `./run.sh --correct "heard" "correct"` (or double-click
**Teach a Word.command**). Teach a Word can also save optional context and app
name, which the local repair model uses as relevant examples later. From the CLI:
`./run.sh --correct "Versal" "Vercel" --correct-context "deploy to Vercel" --correct-app "Cursor"`.

### How corrections are chosen

1. Basic cleanup handles spoken punctuation, fillers, and spacing (unless cleanup
   is `none`).
2. Your saved replacements run once, longest phrase first. A replacement cannot
   trigger another rule or be changed by fuzzy matching. Your chosen spelling
   and capitalization are kept, including names such as `früt`, `iPhone`, and `C++`.
3. English fuzzy repair requires both similar spelling and phonetic evidence,
   with a clear winning candidate. It skips dictionary words, acronyms, URLs,
   email addresses, paths, code, and identifiers. If a real word is a misheard
   name (for example `Versal`), teach that replacement explicitly.
4. Optional local model repair sees the corrected text. Its proposal must consist
   of small, plausible word substitutions; changed numbers, negations, technical
   literals, saved spellings, word counts, or formatting reject the proposal.
   Rejected proposals and model failures retain the deterministic result.

Automatic learning uses a snapshot of the field after insertion. It only compares
that insertion while the surrounding text stays unchanged. Repeated/ambiguous
insertions, document rewrites, and fields over 16,384 characters are skipped.
The same name fix must occur in two separate dictations in the same app before
it becomes a saved correction. Pending observations are stored privately in
`~/.flowdictate/pending_corrections.json`; they do not influence transcription or
repair until confirmed. Existing saved corrections continue to work immediately
and automatic learning cannot overwrite them.

These checks favor keeping your words when uncertain. They reduce specific false
corrections; they cannot guarantee that every speech-recognition error is fixed.

Run the model-free regression suite (requires `numpy`, `rapidfuzz`, `jellyfish`):

```bash
python -m unittest discover -s frutflow/tests -v  # from the repository root
```

`frutflow/try_repair.py` separately exercises the local model on a Mac with GPU
access. Its text examples are a small diagnostic set, not an audio accuracy benchmark.

### Optional on-device cleanup (still 100% local, still free)

There are no cloud back-ends and no API keys anywhere in this app. The one
optional extra is a **fully on-device** cleanup step:

- **On-device misheard-word repair** — a small local model reads the whole
  dictation and fixes clear mishearings (homophones like *there/their*,
  *pier/peer*, or a garbled term the sentence makes obvious). It's conservative:
  it never paraphrases, answers, or touches a word that was already right. No
  cloud, no account, no API key. Everything it needs is already installed by
  `requirements.txt` — just set `"cleanup": "local"` (or pick **On-device** in
  Settings ▸ Model). The repair model (~0.9 GB) downloads once on first use,
  then runs entirely on your Apple-Silicon GPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Hotkey does nothing | Grant **Input Monitoring** to `frutflow` (or the terminal used for a manual launch), then restart it. |
| Nothing gets pasted | Grant **Accessibility**. Try `"insert_method": "type"`. |
| `PortAudioError` / no audio | `brew install portaudio`; check `python flow.py --list-devices`. |
| First run is slow | It's downloading the speech model once; subsequent launches load it from the local cache. |
| First dictation is slow after opening the lid | The current build automatically coalesces closed-lid maintenance wakes, refreshes audio once, and warms the model after a visible wake. Open `~/.flowdictate/flow.log` if this still repeats. |
| Dictation feels slow | The default Parakeet backend is already sub-second/clip. If you switched to `"local"` (faster-whisper), that's the ~2–5 s CPU path — switch back to `"parakeet"`. |
| It keeps misspelling a name | Fix it in two separate dictations to teach it automatically. Or `./run.sh --correct "heard" "correct"`. |
| A real word gets "corrected" | Raise `"fuzzy_threshold"` (e.g. `0.85`), or set `"fuzzy_correct": false`. |
| Old clipboard isn't restored | `restore_clipboard` is on by default; if it misses on a slow Mac, set `"restore_clipboard": false` for dictation-only behavior. |
| Transcription too aggressive | Set `"cleanup": "none"` to keep the raw engine output (your taught corrections still apply). |

## Privacy

**Nothing ever leaves your computer** — with any setting. There are no cloud
back-ends and no API keys anywhere in this app; all transcription and cleanup
runs on-device. First setup can download Python packages and model weights; after
the supported models are cached, dictation itself does not send audio or text to
a server. Your learned vocabulary, corrections, optional history, and logs live
in `~/.flowdictate` and are written owner-only. Live dictation logs redact the
transcript unless `debug` is enabled.

## Credits & licenses

- **früt Flow** is released under the [MIT License](LICENSE).
- **Speech models:** [NVIDIA Parakeet](https://huggingface.co/nvidia) via
  [Apple MLX](https://github.com/ml-explore/mlx) /
  [parakeet-mlx](https://github.com/senstella/parakeet-mlx); optional
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) fallback.
- **Repair model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit)
  (Apache-2.0) via [mlx-lm](https://github.com/ml-explore/mlx-lm), used by the
  optional on-device cleanup.
- **Icons:** [Phosphor Icons](https://phosphoricons.com) — MIT License,
  © Phosphor Icons.
- **Libraries:** rapidfuzz, jellyfish, sounddevice, pynput, numpy, pyobjc.
