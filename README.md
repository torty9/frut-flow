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
- **Automatic accuracy**: phonetic proper‑noun repair + learns corrections from your edits.
- **Menu‑bar app** — no Terminal window needed; self‑contained `frutflow.app`.
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
