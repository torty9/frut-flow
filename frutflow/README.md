# früt Flow

A local, private voice-dictation tool for macOS. Hold a hotkey, speak, release,
and your words are typed into whatever app you're focused on.

Everything runs **on your machine** — Parakeet on the Apple-Silicon GPU by
default. No account, no subscription, no API keys, and no audio or text ever
leaves your computer. It's 100% free.

```
   hold ⌥ (right Option) ─► 🎤 record ─► 🧠 transcribe on-device (Parakeet, GPU)
        ─► ✨ cleanup ─► ✍️ write it in this app's style (optional, on-device)
        ─► 🩹 fix your proper nouns (yours + the ones on screen)
        ─► 📋 paste ─► 👀 watch your edits & learn them automatically
```

## Why it's accurate (the part that stops you editing)

Four layers stack so you rarely have to fix anything — all **on-device, automatic**:

1. **A strong, fast model by default.** NVIDIA **Parakeet** (`parakeet-tdt-0.6b`)
   running on the Apple-Silicon **GPU** via MLX — fast *and* accurate, with
   punctuation and capitalization built in. It renders names as near-misses
   (e.g. *Vercel* → "Versal") rather than unrelated words, which the next layer
   repairs. (`faster-whisper distil-large-v3` is still available as a CPU fallback.)
2. **Phonetic proper-noun repair.** After transcription, near-miss words are
   snapped to your known vocabulary using sound-alike + edit-distance matching
   ("Versal" → "Vercel", "Frut" → "früt"), with guards so real words are never
   touched: an ordinary dictionary word dictated as such ("phone", "call",
   "fruit") is left alone, and three-letter acronyms are never fuzzy targets
   (say "call" with "CLI" in your vocabulary and you still get "call"). Want a
   real word remapped anyway ("fruit" → "früt")? Teach it explicitly — exact
   taught corrections always apply.
3. **Names from the screen in front of you.** The email you're answering already
   says *Vercel*, so früt Flow reads the focused text field and the window title
   and offers their proper nouns to that same repair step for this one dictation —
   "Versal" becomes "Vercel" without anyone teaching it. It's deliberately
   stricter than your own vocabulary: a higher match score, and it only replaces
   a token that isn't an ordinary word ("Austin" is never rewritten to the
   "Austen" on screen). No model involved, so it works in every cleanup mode. The
   text is read through the Accessibility permission paste already needs, used
   in memory, and never stored or logged (`"context_awareness": false` turns it
   off; so does Settings ▸ Privacy ▸ **Use names on screen**).
4. **Automatic learning from your edits.** When you fix a word right after it's
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
pip install -r requirements.lock   # exact, tested-together versions (Python 3.12)
                                   # …or requirements.txt for the ">=" floors
python flow.py --setup
python flow.py
```

`requirements.txt` says *what* früt Flow needs; `requirements.lock` records the
exact versions it is developed against. `run.sh` installs from the lock whenever
the venv's Python matches the one named in the lock's header, so a new install
gets a known-good combination rather than whatever resolves that day.

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

Quit with `Ctrl-C`. (If you installed with `Install frut Flow.command`, the
app also starts at login and a small watchdog relaunches it if it ever dies;
use the menu-bar **Quit** or **`Quit frutflow.command`** to stop it, and drag
`~/Applications/frutflow.app` to the Trash to uninstall — the watchdog stops
on its own once the app is gone.)

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
| `hotkey` | `"alt_r"` | Use Settings ▸ Dictation ▸ **Change**, then press the single key you want: a modifier, a function key, a keypad key. Keys you *type* with (letters, digits, Space, Return, Tab, Delete, arrows) are refused there, because a non-modifier hotkey is consumed system-wide and would stop typing in every app. Regular keys are stored as `vk:N`; modifier names like `alt`, `cmd`, `ctrl`, `shift`, or `cmd_r` also work in JSON — and JSON still accepts any `vk:N` if you really want one. |
| `mode` | `"hold"` | `"hold"` = push-to-talk · `"toggle"` = tap to start/stop |
| `transcribe_backend` | `"parakeet"` | `"parakeet"` (NVIDIA Parakeet on the GPU — fastest, default) · `"local"` (faster-whisper on CPU). Both on-device. |
| `language` | `"en"` | `"en"` · `"es"` · any ISO code · `"auto"` (detect per dictation). Non-English auto-selects a multilingual model. |
| `parakeet_model` | `"mlx-community/parakeet-tdt-0.6b-v2"` | v2 = English (best English accuracy) · `...-v3` = 25 languages incl. Spanish (auto-detects) |
| `model` | `"distil-large-v3"` | faster-whisper model, used when `transcribe_backend: "local"`. English-only; swapped for `large-v3-turbo` automatically when `language` isn't `"en"`. |
| `normalize_method` | `"rms"` | `"rms"` = average-loudness normalize + soft-limit (best for quiet/whispered) · `"peak"` = old peak-normalize |
| `fuzzy_correct` | `true` | Phonetic proper-noun repair against your learned vocab (engine-agnostic) |
| `context_awareness` | `true` | Let that repair also use the names visible in the focused text field and window title (see layer 3 above). Read for one dictation; never stored or logged. |
| `learn_from_edits` | `true` | Auto-learn corrections by watching the field you paste into |
| `cleanup` | `"basic"` | `"none"` (raw) · `"basic"` (fillers, spacing, caps + spoken punctuation like "quote … end quote") · `"local"` (on-device misheard-word repair — no cloud, no key) |
| `style` | `"verbatim"` | How the dictation is written: `"verbatim"` · `"polish"` · `"email"` · `"message"` · `"notes"`. See [Writing styles](#writing-styles-and-app-profiles). |
| `app_profiles` | `[]` | Per-app overrides, e.g. `{"app": "Mail", "bundle_id": "com.apple.mail", "style": "email"}`. See [Writing styles](#writing-styles-and-app-profiles). |
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

You usually won't need to teach it anything — it learns from your edits. But you
still can: Settings ▸ Corrections ▸ **Teach…** (also **Teach a Word…** in the
menu-bar menu), `./run.sh --correct "heard" "correct"`, or double-click
**Teach a Word.command**. Teach a Word can also save optional context and app
name, which the local repair model uses as relevant examples later. From the CLI:
`./run.sh --correct "Versal" "Vercel" --correct-context "deploy to Vercel" --correct-app "Cursor"`.

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
  then runs entirely on your Apple-Silicon GPU. It is loaded in the background
  right after launch, and the unchanging part of its prompt (instructions +
  examples, ~600 tokens) stays cached on the GPU between dictations, so each
  dictation only pays for its own words (~0.3 s saved per dictation).

## Writing styles and app profiles

Dictated speech isn't typed prose: it has false starts, repeated words, and no
layout. A **writing style** hands the dictation to the same on-device model as
the cleanup above and asks it to write the text the way you would have typed it
— still 100% local, no API key.

| Style | What it does |
|---|---|
| `verbatim` | **Default.** Your words as spoken, tidied per `cleanup`. No model. |
| `polish` | Drops false starts and repeated words ("I'll, I'll send it" → "I'll send it"), fixes grammar slips and obvious mishearings. Keeps your wording. |
| `email` | Polish + an email's layout: greeting, short paragraphs, sign-off — **only ones you actually said**, never invented. |
| `message` | Polish for chat: no closing period ("Sounds good"). |
| `notes` | Condenses what you say into `- ` bullets, one per point. Ends on a fresh line, so consecutive dictations build one list. |

Pick the default in Settings ▸ **Apps** ▸ *Writing style*, or — the better way —
**per app**: open Mail, go to Settings ▸ Apps ▸ *Add a running app…*, choose Mail,
and it starts as **Email** (Slack/Messages start as Message, Notes/Obsidian as
Notes; change any of them from its menu). From then on früt Flow switches by
itself according to the app you're dictating into.

**Your words are never at the model's mercy.** Small language models are eager
assistants: asked to tidy the dictated sentence *"what time is the standup
tomorrow?"* they tend to *answer* it. früt Flow counters that twice over. Every
request frames your dictation as quoted material to transcribe, never as a
message; and every rewrite must pass a **faithfulness guard** before it is
typed. The guard refuses a rewrite that

- answers, obeys, translates or comments instead of rewriting,
- introduces a name or a number you didn't say, or drops one you did,
- loses too many of your words or adds new ones,
- turns a question into a statement, or stops addressing "you".

A refused rewrite costs about a second and nothing else: **your exact words are
typed instead** (and `flow.log` names the rule — never your text). When a style
*did* change the text, the History window keeps the words **as spoken** one click
away. Dictations under four words skip the model entirely, a "never mind" is
always judged on what you actually said, and line breaks a style adds are pasted
rather than typed, so they can't press Return in a chat box.

Measured on the default 1.5B model with 21 test dictations per style (ordinary,
long, Spanish, unpunctuated, and seven adversarial ones like "ignore all previous
instructions…"). The model fell for three or four of the adversarial ones in
every style — it translated, it answered, it printed "banana" — and the guard
refused every one of those. In `polish`, `email` and `message`, every rewrite
that reached the cursor said what the speaker said. `notes` is different in
kind: it kept every name and number, but it *condenses*, so it dropped an
acknowledgement ("yeah, that works for me") in one case and the "do you know
if…" framing in another, and the guard refused 7 of 21 where it would have lost
more. Use it where terse is what you want; the words as spoken stay in History.
A style adds roughly 0.2–1 s per dictation (about 2 s for a long one); each
style's prompt stays cached on the GPU.

Try one without speaking:

```bash
./run.sh --try "yeah that works, um, I'll I'll send the deck tonight" --style polish
./run.sh --try "hi tom thanks for the update talk soon henrik" --as-app Mail
./run.sh --try "deploy to Versal failed" --context "Did the deploy to Vercel finish?"
```

### App profiles in `config.json`

A profile can change more than the style. Everything read fresh per dictation is
overridable: `style`, `cleanup`, `insert_method`, `auto_space`,
`restore_clipboard`, `fuzzy_correct`, `context_awareness`, `learn_from_edits`,
`learn_vocab`, `history_enabled`. (The engine, its model and the hotkey are
process-wide and are not.)

```json
"app_profiles": [
  {"app": "Mail",     "bundle_id": "com.apple.mail",            "style": "email"},
  {"app": "Slack",    "bundle_id": "com.tinyspeck.slackmacgap", "style": "message"},
  {"app": "Terminal", "auto_space": false, "cleanup": "none"},
  {"app": "1Password", "history_enabled": false, "learn_vocab": false,
   "learn_from_edits": false, "context_awareness": false}
]
```

An entry with a `bundle_id` matches only that app (display names are localized and
not unique); without one it matches the app's name, case-insensitively. First
match wins; apps with no profile use your global settings unchanged. Settings ▸
Apps edits the style and leaves the other keys alone.

## Troubleshooting

| Symptom | Fix |
|---|---|
| A style didn't change my text | Check `~/.flowdictate/flow.log` for "rewrite was not faithful (…)": the guard refused it and typed your words instead. Under four words, and over `local_repair_max_input_chars`, the model is skipped on purpose. |
| The wrong name got "fixed" from the screen | Turn off Settings ▸ Privacy ▸ **Use names on screen** (or `"context_awareness": false`), globally or for that one app via `app_profiles`. |
| No 🎙️ in the menu bar | Check that `frutflow` is switched on in System Settings ▸ Menu Bar ▸ **Allow in the Menu Bar**. On a MacBook with a notch, icons that don't fit beside it are hidden, so quit a few other menu-bar apps. |
| Hotkey does nothing | Grant **Input Monitoring** to `frutflow` (or the terminal used for a manual launch), then restart it. |
| Nothing gets pasted | Grant **Accessibility**. Try `"insert_method": "type"`. |
| `PortAudioError` / no audio | `brew install portaudio`; check `python flow.py --list-devices`. |
| First run is slow | It's downloading the speech model once; subsequent launches load it from the local cache. |
| First dictation is slow after opening the lid | The current build automatically coalesces closed-lid maintenance wakes, refreshes audio once, and warms the model after a visible wake. Open `~/.flowdictate/flow.log` if this still repeats. |
| Dictation feels slow | The default Parakeet backend is already sub-second/clip. If you switched to `"local"` (faster-whisper), that's the ~2–5 s CPU path — switch back to `"parakeet"`. |
| It keeps misspelling a name | Just fix it once after it pastes — it learns the correction automatically. Or `./run.sh --correct "heard" "correct"`. |
| A real word gets "corrected" | Dictionary words dictated as such are guarded; if a name-like token still snaps wrongly, raise `"fuzzy_threshold"` (e.g. `0.85`) or set `"fuzzy_correct": false`. |
| The first dictation after launch is slow | With `"cleanup": "local"` the repair model is now preloaded right after start-up; if you dictate in those first seconds the clip simply waits for it (watch `flow.log` for "repair model preloaded"). |
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

Two features read text that is already on your screen, both through the
Accessibility permission paste needs anyway and both on-device: *learn from my
edits* re-reads the field you dictated into, and *use names on screen* reads the
focused field and window title to spell names correctly. That text is used in
memory for one dictation and is never written to disk or to the log. Neither
takes screenshots, and früt Flow never asks for Screen Recording. Each has a
switch in Settings ▸ Privacy, and an app profile can turn either off — or keep
an app out of History altogether — for a single app.

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
