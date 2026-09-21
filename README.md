# früt Flow

A local, private, push‑to‑talk **voice dictation** app for macOS. Hold a hotkey,
speak, release, and your words are typed into whatever app you're focused on.
Transcription runs **on‑device** on the Apple‑Silicon GPU (NVIDIA Parakeet via
MLX) — no account, no subscription, no API keys, and no audio ever leaves your
machine. It's **100% free**.

```
hold ⌥ (Right Option) ─► 🎤 record ─► 🧠 transcribe on‑device (Parakeet, GPU)
     ─► ✨ cleanup ─► 🩹 fix your proper nouns ─► 📋 paste ─► 👂 learn your edits
```

It can run right from a Terminal (`./run.sh`) or, once set up, as a real
**menu‑bar app** (`frutflow.app`): a status glyph shows the state
(🎙️ idle · 🔴 listening · ⏳ transcribing), and the menu gives you **Teach a Word,
Restart, Open Log, and Quit**.

## Features

- **On‑device, private** transcription (Parakeet‑MLX on the GPU; `faster‑whisper` fallback).
- **Multilingual** — English, Spanish, and 23 more languages via the Parakeet v3
  model. Pick **Auto / English / Español** in Settings ▸ Model; Auto detects the
  language of each dictation, and the cleanup rules follow (Spanish spoken
  punctuation, `¿ ¡` spacing, Spanish voice undo like *"borra eso"*).
- **Spoken punctuation** — say *"quote … end quote"* for real `"…"` marks,
  *"new paragraph"*, *"question mark"*, or in Spanish *"abrir comillas"*,
  *"punto y aparte"*, *"signo de interrogación"* — with guards so prose like
  "the trial period" or "punto de vista" is never mangled.
- **Automatic accuracy**: phonetic proper‑noun repair + learns corrections from your
  edits — and spells names the way the **text on screen** already does ("Versal" →
  "Vercel" when Vercel is in the email you're answering), with nothing stored.
- **Writing styles, per app, fully offline** — *Polish* drops false starts and
  repeated words, *Email* adds greeting/paragraphs/sign‑off, *Message* is chat‑ready,
  *Notes* makes bullets. Pick one per app in Settings ▸ Apps (Email in Mail,
  Message in Slack…) and früt Flow switches by itself. It runs on the same
  on‑device model as cleanup — no API key — behind a **faithfulness guard**: if
  the model adds, drops or *answers* anything, your exact words are typed instead.
- **Menu‑bar app** — no Terminal window needed; `frutflow.app` owns the macOS
  permissions while the code and venv live beside it for easy updates.
- **Voice undo — "never mind."** If a whole utterance is *"never mind"* (or *"scratch
  that"*, *"actually never mind"*, …), the previous dictation is deleted instead of
  typed — but only while your cursor is still in the app you dictated into.
  Configurable in `~/.flowdictate/config.json` (`undo_phrases`).

## Layout

| Path | What it is |
|---|---|
| [`frutflow/flow.py`](frutflow/flow.py) | The whole app (transcription, hotkey, paste, learning, menu bar). |
| [`frutflow/README.md`](frutflow/README.md) | Full setup, permissions, and configuration docs. |
| `frutflow/*.command` / `*.sh` | Launchers, the self‑heal watchdog, and helpers. |
| `frutflow/config.example.json` | Template for `~/.flowdictate/config.json`. |

> The app runs from an in‑place virtualenv (`frutflow/.venv`, not committed). See
> [`frutflow/README.md`](frutflow/README.md) for install, the one‑time macOS
> permission grants (Microphone, Input Monitoring, Accessibility), and configuration.

## License

[MIT](LICENSE) — free to use, modify, and share.
