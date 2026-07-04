#!/usr/bin/env python3
"""
Wispr DIY — a local, private voice-dictation tool for macOS.

Hold a hotkey, speak, release. Your speech is transcribed *on your machine*
and pasted into whatever app you're focused on. No subscription, no account,
and (by default) no audio ever leaves your computer.

This is a from-scratch alternative to Wispr Flow. It reproduces the core loop:
    push-to-talk hotkey -> capture mic audio -> speech-to-text -> light cleanup
    -> insert text at the cursor of the active app.

Run it:           python3 flow.py
First-time setup: python3 flow.py --setup   (writes ~/.flowdictate/config.json
                  and prints the macOS permissions you need to grant)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# We load the Parakeet weights from the local cache, so the fast xet transfer path
# is never needed; disable it (harmless, and avoids one class of Hub network call).
# NOTE: huggingface_hub may still print a cosmetic "sending unauthenticated requests
# to the HF Hub" advisory to stderr — it's a benign metadata check, NOT your audio;
# nothing about dictation leaves the machine. Set before any HF import.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".flowdictate"
CONFIG_PATH = CONFIG_DIR / "config.json"

SAMPLE_RATE = 16_000   # Whisper expects 16 kHz
CHANNELS = 1

DEFAULT_CONFIG = {
    # --- activation ---
    "hotkey": "alt_r",           # push-to-talk key. A pynput Key name (e.g.
                                 # "alt_r", "alt_l", "cmd_r", "ctrl_r", "f6")
                                 # or a single character. Use a MODIFIER key so
                                 # holding it doesn't type into your document.
    "mode": "hold",              # "hold"  = push-to-talk (hold while speaking)
                                 # "toggle"= tap to start, tap again to stop
    "debug": False,              # when true, log repr(key)+vk for EVERY key event
                                 # so real human keypresses are visible in flow.log

    # --- transcription ---
    "transcribe_backend": "parakeet",  # "parakeet" (NVIDIA Parakeet via MLX,
                                 # runs on the M-series GPU — DEFAULT, ~6-12x
                                 # faster than faster-whisper AND more accurate,
                                 # with native punctuation/caps), "local"
                                 # (faster-whisper on CPU), or "openai" (cloud).
                                 # If the parakeet backend fails to load for any
                                 # reason we fall back to "local" automatically.
    "parakeet_model": "mlx-community/parakeet-tdt-0.6b-v2",
                                 # Parakeet checkpoint. v2 = English-only, best
                                 # English WER (6.05% on the Open ASR leaderboard).
                                 # Use "mlx-community/parakeet-tdt-0.6b-v3" for 25
                                 # languages (slightly worse English).
    "warmup_on_start": True,     # run one throwaway inference at startup so the
                                 # FIRST real dictation isn't slowed by ~2.8s of
                                 # Metal-kernel compilation (Parakeet only).
    "model": "distil-large-v3",  # faster-whisper model, used when
                                 # transcribe_backend == "local". ACCURACY-FIRST.
                                 # Research + live A/B on this M4 showed the small
                                 # models (base.en/small.en) routinely mangle proper
                                 # nouns ("Vercel"->"versatile", "früt"->"front"),
                                 # which is the #1 cause of post-dictation editing.
                                 # distil-large-v3 instead renders them as near-miss
                                 # NAMES ("Versal"/"Frut") that the fuzzy corrector
                                 # below snaps to the right spelling. Cost: ~2-5s per
                                 # clip vs ~0.5-2s. Want it snappier? Drop to
                                 # "medium.en" or "small.en" (less accurate). Want
                                 # the max-accuracy ceiling? keep this.
    "compute_type": "int8",      # "int8" (CPU, default) or "float16" (GPU)
    "cpu_threads": 0,            # CPU threads for transcription. 0 = auto: use all
                                 # of this Mac's cores. Measured ~28% faster than
                                 # CTranslate2's stock default on an M4 (6.2s->4.4s)
                                 # with byte-identical output — a free speedup. Set a
                                 # smaller number to leave more headroom for other
                                 # apps while dictating.
    "language": "en",

    # --- accuracy tuning (esp. for quiet / whispered speech) ---
    "normalize_audio": True,     # normalize the clip's loudness before transcribing
                                 # so quiet/whispered input is boosted to a
                                 # consistent level. Biggest near-free accuracy win.
    "normalize_method": "rms",   # "rms"  = target average loudness (RMS), then
                                 #          soft-limit peaks. Best for QUIET/whispered
                                 #          speech: a single loud key-click no longer
                                 #          starves the boost the way peak-norm does.
                                 # "peak" = scale so the single loudest sample hits
                                 #          normalize_peak (the older behavior).
    "normalize_rms_dbfs": -20.0, # target average level for "rms" mode (dBFS). Higher
                                 # (e.g. -18) = louder/more boost; lower = gentler.
    "normalize_peak": 0.95,      # peak target for "peak" mode AND the soft-limit
                                 # ceiling for "rms" mode (~ -0.5 dBFS).
    "vad_filter": False,         # push-to-talk gives exact speech boundaries, so
                                 # VAD only risks trimming low-energy (whispered)
                                 # words. Off by default; set True to re-enable.
    "beam_size": 5,              # accuracy/compute sweet spot (faster-whisper default)
    "initial_prompt": "",        # optional vocab primer, e.g. "früt, frut.energy,
                                 # co-packer, Vercel, GitHub, Austin, Texas." to bias
                                 # spelling of proper nouns. Empty = no prompt.
    "vocab_biasing": "hotwords", # how to bias the model toward YOUR words:
                                 #   "hotwords"  = faster-whisper's dedicated hotwords
                                 #                 param (a short, curated list — the
                                 #                 reliable lever; default).
                                 #   "prompt"    = the old initial_prompt blob (weaker,
                                 #                 hallucination-prone; kept for compat).
                                 #   "off"       = no decoder biasing (the fuzzy
                                 #                 corrector below still fixes names).
    "learn_vocab": True,         # learn your words as you use it: each dictation
                                 # updates ~/.flowdictate/vocab.json and biases
                                 # future transcriptions toward your vocabulary.

    # --- post-processing ---
    "cleanup": "basic",          # "none" | "basic" | "llm"  (llm uses Anthropic)
    "llm_model": "claude-haiku-4-5-20251001",
    "fuzzy_correct": True,       # AUTOMATIC, on-device proper-noun repair: snap
                                 # near-miss tokens ("Versal"->"Vercel", "Frut"->"früt")
                                 # to your known vocabulary using phonetic + edit-
                                 # distance matching (case-preserving, guarded against
                                 # false positives). This is the layer that fixes the
                                 # names the model still gets slightly wrong.
    "fuzzy_threshold": 0.74,     # 0..~1.25 match score floor. Higher = stricter
                                 # (fewer corrections); lower = more aggressive.
    "learn_from_edits": True,    # THE no-manual-teaching loop: after pasting, watch
                                 # the field you typed into; if you fix a word, learn
                                 # that correction automatically (phonetically gated to
                                 # real mis-hears, restricted to proper nouns). Kills
                                 # the need to ever run `--correct` by hand.
    "learn_window_seconds": 20,  # how long after a paste to keep watching for your
                                 # edit before finalizing what was learned.

    # --- text insertion ---
    "insert_method": "paste",    # "paste" (clipboard + Cmd-V), "type", or
                                 # "clipboard" (copy only, no Accessibility)
    "restore_clipboard": True,   # put your old clipboard back after pasting
    "auto_space": True,          # prepend a space so dictation merges naturally
                                 # with text already in the field (Wispr-style)

    # --- voice undo ("never mind") ---
    "undo_enabled": True,        # if a whole dictation is just an undo phrase (below),
                                 # delete the PREVIOUS dictation instead of typing it.
                                 # e.g. say "the dog ran outside", then "never mind" ->
                                 # the last sentence is backspaced away. Say it again to
                                 # peel off the one before, and so on.
    "undo_phrases": [            # exact whole-utterance triggers (case/punctuation-insensitive)
        "never mind", "nevermind", "actually never mind", "actually nevermind",
        "scratch that", "actually scratch that", "delete that", "actually delete that",
        "cancel that", "forget that", "undo that",
    ],

    # --- feedback / guards ---
    "play_sounds": True,
    "show_hud": True,            # floating waveform pill near the bottom of the screen
                                 # while you dictate (menu-bar/app mode only). Cosmetic.
    "ding_volume": 0.25,         # volume (0.0–1.0) of the success ding; lower = quieter
    "min_seconds": 0.2,          # ignore accidental sub-200ms taps
    "max_record_seconds": 120,   # safety cap: auto-stop a capture this long (guards a
                                 # missed key-up). A warning sound plays shortly before.
    "warn_before_max_seconds": 10,  # play a distinct warning this many seconds before
                                 # the cap, so a long dictation isn't cut off by surprise.
    "max_processing_seconds": 120,  # if transcription/paste is somehow still running
                                 # after this long, log loudly (a stuck worker used to
                                 # fail silently -> "dead hotkey").
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text())
            cfg.update({k: v for k, v in user.items() if k in DEFAULT_CONFIG})
        except (json.JSONDecodeError, OSError) as e:
            print(f"[flow] WARNING: could not read {CONFIG_PATH}: {e}")
    return cfg


def write_default_config() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        print(f"[flow] config already exists at {CONFIG_PATH} (leaving it as-is)")
        return
    CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
    print(f"[flow] wrote default config to {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------

class Recorder:
    """Records mono 16 kHz float32 audio between start() and stop()."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        import sounddevice as sd  # imported lazily so --setup works without it
        self._sd = sd
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()
        self.recording = False

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        # status carries xrun warnings; we just keep grabbing audio.
        self._frames.append(indata.copy())

    def _open_stream(self):
        stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        stream.start()
        return stream

    def start(self) -> None:
        with self._lock:
            if self.recording:
                return
            self._frames = []
            try:
                self._stream = self._open_stream()
            except Exception:  # noqa: BLE001
                # After sleep/wake, PortAudio's cached device state can be stale
                # (or the default input changed while we slept — AirPods etc.).
                # Reinitialize the library once and retry before giving up.
                try:
                    self._sd._terminate()
                    self._sd._initialize()
                except Exception:  # noqa: BLE001
                    pass
                self._stream = self._open_stream()   # raises to caller if still bad
            self.recording = True

    def stop(self) -> np.ndarray | None:
        with self._lock:
            if not self.recording:
                return None
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
                self.recording = False
            if not self._frames:
                return None
            return np.concatenate(self._frames, axis=0).flatten()


# ---------------------------------------------------------------------------
# Transcription back-ends
# ---------------------------------------------------------------------------

def normalize_audio(audio: np.ndarray, method: str = "rms",
                    peak: float = 0.95, rms_dbfs: float = -20.0) -> np.ndarray:
    """Loudness-normalize a float32 clip so quiet/whispered speech is boosted to a
    consistent level before transcription, then keep it in the expected [-1, 1] range.

    method="peak" (the original behavior): scale so the single loudest sample hits
        `peak`. Simple, but a lone transient — a key click, a lip smack — becomes the
        peak and *starves* the boost: the actual (quiet) speech stays quiet.

    method="rms" (default): scale to a target AVERAGE level (`rms_dbfs`), which tracks
        how loud the speech really is rather than its loudest spike, then pass the
        result through a tanh SOFT-LIMITER at `peak`. tanh is ~linear well below the
        ceiling (so quiet speech is untouched and undistorted) and saturates smoothly
        above it (so the boosted key-click is tamed instead of hard-clipped). This is
        the research-backed win for whispered/quiet dictation.

    A near-silence guard avoids amplifying a pure-silence buffer up to full scale
    (which would just raise the noise floor and invite hallucination), and the gain is
    capped (+40 dB) so a near-silent buffer can't be blown up arbitrarily.
    """
    if audio is None or audio.size == 0:
        return audio
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    if method == "peak":
        pk = float(np.max(np.abs(audio)))
        if pk > 1e-4:
            audio = audio * (peak / pk)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    # --- rms (default) ---
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms > 1e-5:                       # near-silence guard
        target_rms = 10.0 ** (rms_dbfs / 20.0)
        gain = min(target_rms / rms, 100.0)   # cap boost at +40 dB
        audio = audio * gain
    # Soft-limit peaks toward the ceiling (gentle compression + limiter in one).
    ceil = max(peak, 1e-3)
    audio = ceil * np.tanh(audio / ceil)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


class LocalTranscriber:
    """On-device speech-to-text via faster-whisper. Nothing leaves the Mac."""

    def __init__(self, model_name: str, compute_type: str, language: str,
                 *, normalize: bool = True, normalize_peak: float = 0.95,
                 normalize_method: str = "rms", normalize_rms_dbfs: float = -20.0,
                 vad_filter: bool = False, beam_size: int = 5,
                 initial_prompt: str = "", cpu_threads: int = 0):
        from faster_whisper import WhisperModel
        self.language = language
        self.normalize = normalize
        self.normalize_peak = normalize_peak
        self.normalize_method = normalize_method
        self.normalize_rms_dbfs = normalize_rms_dbfs
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        # faster-whisper treats an empty initial_prompt as "no prompt" (None).
        self.initial_prompt = initial_prompt or None
        # 0 = use every core. CTranslate2's stock auto-pick leaves the M4 cores
        # underused; passing the full count is a free ~28% speedup, same output.
        import os
        threads = cpu_threads if cpu_threads and cpu_threads > 0 else (os.cpu_count() or 4)
        print(f"[flow] loading local model '{model_name}' "
              f"({compute_type}, {threads} threads) ...", file=sys.stderr, flush=True)
        # First run downloads the model from Hugging Face, then it's cached.
        self.model = WhisperModel(model_name, compute_type=compute_type,
                                  cpu_threads=threads)
        print("[flow] model ready.", file=sys.stderr, flush=True)

    def transcribe(self, audio: np.ndarray, prompt: str | None = None,
                   hotwords: str | None = None) -> str:
        if self.normalize:
            audio = normalize_audio(audio, method=self.normalize_method,
                                    peak=self.normalize_peak,
                                    rms_dbfs=self.normalize_rms_dbfs)
        else:
            audio = np.ascontiguousarray(audio, dtype=np.float32)
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            # Push-to-talk gives exact boundaries; VAD off by default so quiet
            # (low-energy) speech isn't trimmed. Configurable via "vad_filter".
            vad_filter=self.vad_filter,
            # Each utterance is independent — don't feed a prior transcript back
            # in (default True causes repetition / hallucination drift).
            condition_on_previous_text=False,
            # Keep the full temperature fallback ladder (faster-whisper default),
            # set explicitly so it can't be accidentally pinned to greedy-only.
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            # Anti-hallucination safety net (faster-whisper defaults), kept
            # explicit now that normalization can raise the noise floor.
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            # Two ways to bias spelling of proper nouns / jargon toward YOUR
            # vocabulary. `hotwords` (added in faster-whisper 1.0.2) is the
            # dedicated, more reliable lever — a short curated list encoded after
            # the SOT-prev token. `initial_prompt` is the older, weaker, more
            # hallucination-prone path. The caller picks ONE via "vocab_biasing";
            # they share a ~224-token budget so we never send both.
            hotwords=(hotwords or None),
            initial_prompt=(prompt or (None if hotwords else self.initial_prompt)),
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class OpenAITranscriber:
    """Optional cloud back-end (Whisper API). Needs OPENAI_API_KEY."""

    def __init__(self, language: str, *, normalize: bool = True,
                 normalize_peak: float = 0.95, normalize_method: str = "rms",
                 normalize_rms_dbfs: float = -20.0):
        from openai import OpenAI
        self.client = OpenAI()
        self.language = language
        self.normalize = normalize
        self.normalize_peak = normalize_peak
        self.normalize_method = normalize_method
        self.normalize_rms_dbfs = normalize_rms_dbfs

    def transcribe(self, audio: np.ndarray, prompt: str | None = None,
                   hotwords: str | None = None) -> str:
        import io
        import wave
        # whisper-1 only exposes `prompt`, so fold any hotwords into it.
        prompt = " ".join(p for p in (prompt, hotwords) if p) or None
        if self.normalize:
            audio = normalize_audio(audio, method=self.normalize_method,
                                    peak=self.normalize_peak,
                                    rms_dbfs=self.normalize_rms_dbfs)
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        buf.seek(0)
        buf.name = "audio.wav"
        resp = self.client.audio.transcriptions.create(
            model="whisper-1", file=buf, language=self.language,
            prompt=(prompt or None),
        )
        return resp.text.strip()


def _hf_repo_cached(repo_id: str) -> bool:
    """True if a Hugging Face repo already has a local snapshot, so we can load it
    fully offline (no network round-trip / no hang when disconnected)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        base = Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        base = Path.home() / ".cache" / "huggingface" / "hub"
    folder = "models--" + repo_id.replace("/", "--")
    snap = base / folder / "snapshots"
    try:
        return snap.is_dir() and any(snap.iterdir())
    except OSError:
        return False


class ParakeetTranscriber:
    """On-device speech-to-text via NVIDIA Parakeet (parakeet-mlx), running on the
    Apple-Silicon GPU through MLX. Nothing leaves the Mac.

    Much faster than faster-whisper on this hardware (CTranslate2 is CPU-only on
    Apple Silicon) and at least as accurate, with punctuation + capitalization built
    in. It has NO decoder biasing (no `hotwords`/`prompt`), so those args are accepted
    for interface parity but ignored — proper-noun personalization is handled entirely
    by the engine-agnostic fuzzy/exact correction layer downstream.
    """

    #: clips shorter than this are padded with trailing silence, because the upstream
    #: Parakeet models can drop very short (<~1s) utterances.
    _MIN_AUDIO_SECONDS = 1.0

    def __init__(self, model_name: str, language: str, *, normalize: bool = True,
                 normalize_peak: float = 0.95, normalize_method: str = "rms",
                 normalize_rms_dbfs: float = -20.0, warmup: bool = True):
        import logging
        # (HF_HUB_DISABLE_XET is set at module top, before HF is imported.)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        from parakeet_mlx import from_pretrained
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel
        self._mx = mx
        self._get_logmel = get_logmel
        self.language = language
        self.normalize = normalize
        self.normalize_peak = normalize_peak
        self.normalize_method = normalize_method
        self.normalize_rms_dbfs = normalize_rms_dbfs

        # Load fully offline when the weights are already cached, so startup never
        # blocks on the network; otherwise allow the one-time download.
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        if _hf_repo_cached(model_name):
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            print("[flow] first run: downloading Parakeet weights "
                  "(~2.3 GB, one time)...", file=sys.stderr, flush=True)
        print(f"[flow] loading Parakeet model '{model_name}' (MLX/GPU) ...",
              file=sys.stderr, flush=True)
        try:
            self.model = from_pretrained(model_name)
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline

        # A real inference on the MAIN thread here is REQUIRED, not merely an
        # optimization. It does two things: (1) compiles the Metal kernels so the
        # first real dictation isn't slowed by ~2s, and (2) initializes MLX's default
        # GPU stream so the transcription WORKER THREAD can use it — a fresh thread
        # otherwise raises "no Stream(gpu, 0) in current thread" on the first decode.
        # It must run on NON-SILENT audio: a silent clip short-circuits before doing
        # any GPU work and does NOT initialize the stream (verified), so we feed a
        # faint noise clip. This always runs; `warmup` only controls the log line.
        try:
            t0 = time.monotonic()
            rng = np.random.default_rng(0)
            self.transcribe((rng.standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32))
            if warmup:
                print(f"[flow] model ready. (warm-up {time.monotonic()-t0:.1f}s)",
                      file=sys.stderr, flush=True)
            else:
                print("[flow] model ready.", file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[flow] model ready. (warm-up issue: {e})",
                  file=sys.stderr, flush=True)

    def transcribe(self, audio: np.ndarray, prompt: str | None = None,
                   hotwords: str | None = None) -> str:
        # prompt/hotwords intentionally ignored — Parakeet has no decoder biasing.
        if audio is None or audio.size == 0:
            return ""
        if self.normalize:
            audio = normalize_audio(audio, method=self.normalize_method,
                                    peak=self.normalize_peak,
                                    rms_dbfs=self.normalize_rms_dbfs)
        else:
            audio = np.ascontiguousarray(audio, dtype=np.float32)
        # Pad very short clips so the model doesn't drop them.
        min_len = int(self._MIN_AUDIO_SECONDS * SAMPLE_RATE)
        if audio.shape[0] < min_len:
            audio = np.concatenate(
                [audio, np.zeros(min_len - audio.shape[0], dtype=np.float32)])
        mel = self._get_logmel(self._mx.array(audio), self.model.preprocessor_config)
        result = self.model.generate(mel)[0]
        return (result.text or "").strip()


def build_transcriber(cfg: dict):
    backend = cfg.get("transcribe_backend", "parakeet")
    norm_kwargs = dict(
        normalize=cfg.get("normalize_audio", True),
        normalize_peak=cfg.get("normalize_peak", 0.95),
        normalize_method=cfg.get("normalize_method", "rms"),
        normalize_rms_dbfs=cfg.get("normalize_rms_dbfs", -20.0),
    )

    if backend == "openai":
        return OpenAITranscriber(cfg["language"], **norm_kwargs)

    if backend == "parakeet":
        try:
            return ParakeetTranscriber(
                cfg.get("parakeet_model", "mlx-community/parakeet-tdt-0.6b-v2"),
                cfg["language"], warmup=cfg.get("warmup_on_start", True),
                **norm_kwargs)
        except Exception as e:  # noqa: BLE001  never leave the user with no engine
            print(f"[flow] WARNING: could not start Parakeet backend ({e}); "
                  "falling back to faster-whisper ('local').", flush=True)

    return LocalTranscriber(
        cfg["model"], cfg["compute_type"], cfg["language"],
        vad_filter=cfg.get("vad_filter", False),
        beam_size=cfg.get("beam_size", 5),
        initial_prompt=cfg.get("initial_prompt", ""),
        cpu_threads=cfg.get("cpu_threads", 0),
        **norm_kwargs,
    )


# ---------------------------------------------------------------------------
# Personalization — learns your vocabulary, and applies corrections you teach it
# ---------------------------------------------------------------------------
#
# Two lightweight, fully on-device learning loops (NO model retraining):
#   1) Vocabulary — every finished dictation updates a word-frequency map; your
#      most-used distinctive words are fed back as Whisper's initial_prompt, so
#      it increasingly nails your names / jargon / brands. Improves with use.
#   2) Corrections — a "misheard -> correct" map applied after transcription.
#      Add one with:  python flow.py --correct "heard" "correct"

VOCAB_PATH = CONFIG_DIR / "vocab.json"
CORRECTIONS_PATH = CONFIG_DIR / "corrections.json"

# Ultra-common words add no value to the vocab prompt — skip them so only your
# distinctive vocabulary is learned.
_COMMON_WORDS = frozenset("""
the a an and or but if then else of to in on at by for with from into over under
is are was were be been being am do does did done have has had having get got
will would can could should shall may might must i you he she it we they me him
her us them my your his its our their this that these those there here what which
who whom not no yes so as than too very just only also more most much many few
some any about after again all because before how out up down off once each
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return default


def learn_vocab(text: str) -> None:
    """Fold a finished dictation into the rolling word-frequency map."""
    if not text:
        return
    vocab = _read_json(VOCAB_PATH, {})
    if not isinstance(vocab, dict):
        vocab = {}
    for w in _WORD_RE.findall(text):
        lw = w.lower()
        if lw in _COMMON_WORDS:
            continue
        entry = vocab.get(lw) or {"count": 0, "form": w}
        entry["count"] = int(entry.get("count", 0)) + 1
        # Prefer a capitalized surface form (likely a proper-noun spelling).
        if w[:1].isupper() and not str(entry.get("form", ""))[:1].isupper():
            entry["form"] = w
        vocab[lw] = entry
    # Bound the file: keep the 400 most-used entries.
    if len(vocab) > 400:
        vocab = dict(sorted(vocab.items(),
                            key=lambda kv: kv[1].get("count", 0),
                            reverse=True)[:400])
    try:
        VOCAB_PATH.write_text(json.dumps(vocab))
    except OSError:
        pass


def load_corrections() -> dict:
    """{'misheard': 'correct'} — whole-word, case-insensitive swaps."""
    data = _read_json(CORRECTIONS_PATH, {})
    return data if isinstance(data, dict) else {}


def add_correction(heard: str, correct: str, *, silent: bool = False) -> None:
    if not heard or not correct or heard == correct:
        return
    corr = load_corrections()
    corr[heard] = correct
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_PATH.write_text(json.dumps(corr, indent=2) + "\n")
    if not silent:
        print(f"[flow] correction saved: '{heard}' -> '{correct}'  ({CORRECTIONS_PATH})")
        print("[flow] (takes effect on your next dictation — no restart needed)")


def apply_corrections(text: str) -> str:
    if not text:
        return text
    for heard, correct in load_corrections().items():
        if heard:
            text = re.sub(rf"\b{re.escape(heard)}\b", correct, text,
                          flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Dictation history  (~/.flowdictate/history.json)
#
# Pure-Python, AppKit-free, worker-thread-safe. Every finalized dictation is
# appended best-effort; a failure here must NEVER propagate into the paste path.
# Entry shape: {"text": str, "ts": float, "app": str|None, "words": int,
#               "delivered": bool}. Stored OLDEST-first on disk (cheap append +
# slice cap); load_history() returns NEWEST-first for the UI. Capped at the last
# HISTORY_CAP entries — your last hundred dictations.
# ---------------------------------------------------------------------------
HISTORY_PATH = CONFIG_DIR / "history.json"
HISTORY_CAP = 100
_HISTORY_LOCK = threading.Lock()   # serialize worker append vs. clear vs. itself


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON so a concurrent reader never sees a torn file: serialize to a
    temp file IN THE SAME DIRECTORY (so os.replace is a same-filesystem atomic
    rename — a cross-device replace would raise), flush+fsync, then replace.
    Caller holds _HISTORY_LOCK. Re-raises on failure (its only callers guard it)."""
    import tempfile
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # atomic on the same filesystem; no torn reads
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_history() -> list:
    """Return the dictation history, NEWEST-FIRST (index 0 is most recent).
    Never raises: a missing or corrupt file yields []. Because the file is only
    ever swapped in atomically, a lock-free read can never see a partial write.
    Defensively drops any entry that isn't a well-formed dict with text."""
    data = _read_json(HISTORY_PATH, [])
    if not isinstance(data, list):
        return []
    good = [e for e in data if isinstance(e, dict) and isinstance(e.get("text"), str)]
    return list(reversed(good))        # disk oldest-first -> newest-first for UI


def record_history(text: str, app: "str | None" = None, delivered: bool = True) -> None:
    """Append one finalized dictation (best-effort, thread-safe, atomic, capped).
    NEVER raises into the caller: runs on the dictation worker thread and must
    not be able to break a paste. Any failure is swallowed."""
    try:
        text = (text or "").strip()
        if not text:
            return
        entry = {
            "text": text,
            "ts": time.time(),
            "app": app,
            "words": len(text.split()),
            "delivered": bool(delivered),
        }
        with _HISTORY_LOCK:
            data = _read_json(HISTORY_PATH, [])
            if not isinstance(data, list):
                data = []
            data.append(entry)                  # oldest-first on disk
            if len(data) > HISTORY_CAP:
                data = data[-HISTORY_CAP:]       # keep the LAST N
            _atomic_write_json(HISTORY_PATH, data)
    except Exception:  # noqa: BLE001  history is a nicety; never break dictation
        pass


def clear_history() -> None:
    """Empty the history atomically (write [] rather than unlink, so a reader
    mid-flight still sees a valid file). Best-effort; never raises."""
    try:
        with _HISTORY_LOCK:
            _atomic_write_json(HISTORY_PATH, [])
    except Exception:  # noqa: BLE001
        pass


def pop_history_matching(text: str) -> None:
    """Remove the NEWEST history entry whose text matches this just-retracted
    dictation — a 'never mind' backspaced it out of the target app, so it should
    no longer appear in History. Matched by CONTENT (not position) because the
    history log and the undo stack can diverge: clipboard-only and error-path
    dictations are recorded but never pushed onto the undo stack. Best-effort;
    never raises (called from the dictation worker thread)."""
    try:
        t = (text or "").strip()
        if not t:
            return
        with _HISTORY_LOCK:
            data = _read_json(HISTORY_PATH, [])
            if not isinstance(data, list):
                return
            for i in range(len(data) - 1, -1, -1):   # disk is oldest-first; scan newest
                e = data[i]
                if isinstance(e, dict) and str(e.get("text", "")).strip() == t:
                    del data[i]
                    _atomic_write_json(HISTORY_PATH, data)
                    return
    except Exception:  # noqa: BLE001
        pass


def relative_time(ts, now: "float | None" = None) -> str:
    """iOS-style relative label: 'just now' / '2m ago' / '3h ago' / 'yesterday'
    / '4d ago' / a short date. Pure; trivially unit-testable. Never raises."""
    try:
        now = time.time() if now is None else float(now)
        d = now - float(ts)
        if d < 0:
            d = 0.0
        if d < 45:
            return "just now"
        if d < 3600:
            return f"{int(round(d / 60)) or 1}m ago"
        if d < 86400:
            return f"{int(d // 3600)}h ago"
        # Calendar-aware day bucketing so "yesterday" means the prior calendar day.
        a = time.localtime(now)
        b = time.localtime(ts)

        def _midnight(t):
            return time.mktime((t.tm_year, t.tm_mon, t.tm_mday,
                                0, 0, 0, 0, 0, -1))
        day = int(round((_midnight(a) - _midnight(b)) / 86400))
        if day <= 1:
            return "yesterday"
        if day < 7:
            return f"{day}d ago"
        return time.strftime("%b %-d", b)
    except Exception:  # noqa: BLE001
        return ""


def build_learned_prompt(cfg: dict) -> str:
    """Whisper initial_prompt biased toward YOUR words: config primer +
    correction targets + most-used distinctive vocabulary (length-capped).
    Used only when vocab_biasing == "prompt" (the weaker, legacy path)."""
    parts = []
    base = (cfg.get("initial_prompt") or "").strip()
    if base:
        parts.append(base)
    parts.extend(sorted({v for v in load_corrections().values() if v}))
    vocab = _read_json(VOCAB_PATH, {})
    if isinstance(vocab, dict):
        ranked = sorted(vocab.values(),
                        key=lambda e: e.get("count", 0), reverse=True)
        learned = [e["form"] for e in ranked
                   if e.get("form") and e.get("count", 0) >= 2][:40]
        parts.extend(learned)
    return " ".join(parts).strip()[:600]


# A real English wordlist (macOS ships one) lets us tell genuine proper nouns /
# brands / jargon from ordinary words that merely got sentence-initial capitals
# ("Use", "Make", "Look"). This matters a lot: those common words must NOT become
# hotwords or fuzzy-correction targets, or we'd corrupt "Mike"->"Make" etc.
_ENGLISH_WORDS: frozenset[str] | None = None


def _english_words() -> frozenset[str]:
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        for p in ("/usr/share/dict/words", "/usr/share/dict/web2"):
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    _ENGLISH_WORDS = frozenset(
                        w.strip().lower() for w in f if w.strip())
                    break
            except OSError:
                continue
        if _ENGLISH_WORDS is None:
            _ENGLISH_WORDS = frozenset()
    return _ENGLISH_WORDS


def _is_distinctive(form: str) -> bool:
    """True if `form` looks like a name/brand/jargon worth biasing toward and
    fuzzy-correcting to — and NOT an ordinary English word."""
    if not form or len(form) < 2:
        return False
    if "'" in form or "’" in form:
        return False   # contractions (I'm, There's) are never proper nouns
    # Strong signals: non-ASCII letter (früt), digit, internal . - _ /, an
    # ALL-CAPS acronym (SPF, LLC), or internal capitals (GitHub, iPhone).
    if any(ord(c) > 127 or c.isdigit() or c in "-._/" for c in form):
        return True
    if form.isupper():
        return True
    if any(c.isupper() for c in form[1:]):
        return True
    low = form.lower()
    if low in _COMMON_WORDS:
        return False
    # A capitalized token that isn't a known English word == likely a name/brand.
    if form[:1].isupper():
        words = _english_words()
        return low not in words if words else True
    return False


def distinctive_terms(max_terms: int = 60) -> list[str]:
    """Your canonical proper-noun / jargon vocabulary, most-used first.

    Correction TARGETS are always canonical (you fixed them on purpose); learned
    vocab is included only if it's used (count>=2), distinctive, and not a common
    word. This is the term list used BOTH for hotword biasing and for the fuzzy
    proper-noun corrector, so the two stay in sync automatically.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for v in load_corrections().values():
        if v and v.lower() not in seen:
            seen.add(v.lower())
            terms.append(v)
    vocab = _read_json(VOCAB_PATH, {})
    if isinstance(vocab, dict):
        ranked = sorted(vocab.values(),
                        key=lambda e: e.get("count", 0), reverse=True)
        for e in ranked:
            form = e.get("form")
            if not form or e.get("count", 0) < 2:
                continue
            low = form.lower()
            if low in seen or low in _COMMON_WORDS or not _is_distinctive(form):
                continue
            seen.add(low)
            terms.append(form)
            if len(terms) >= max_terms:
                break
    return terms


def build_hotwords(cfg: dict) -> str:
    """A SHORT, curated, space-separated hotword string for faster-whisper.

    Unlike the old 600-char initial_prompt blob, this stays well under the
    ~111-token hotword budget and contains only distinctive terms, so it biases
    spelling without inviting hallucination."""
    base = (cfg.get("initial_prompt") or "").strip()
    extra = [t for t in re.split(r"[,\s]+", base) if t]
    out: list[str] = []
    seen: set[str] = set()
    for t in extra + distinctive_terms(max_terms=48):
        k = t.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return " ".join(out)[:400]


# ---------------------------------------------------------------------------
# Phonetic + fuzzy proper-noun repair (engine-agnostic; the reliable layer)
# ---------------------------------------------------------------------------
#
# The most reliable way to fix a misheard proper noun is not to coax the decoder
# but to snap the output token to a known-correct spelling afterwards. We match
# each word against your distinctive vocabulary using BOTH edit distance
# (rapidfuzz) and phonetic codes (jellyfish metaphone/soundex), with a length
# guard and case preservation: "Versal"->"Vercel", "Frut"->"früt", while
# "versatile" / "Boston" are left untouched. Validated on real distil-large-v3
# output. No-ops gracefully if the libraries aren't installed.

try:
    import jellyfish as _jellyfish
except Exception:  # noqa: BLE001
    _jellyfish = None
try:
    from rapidfuzz.distance import Levenshtein as _RFLev
except Exception:  # noqa: BLE001
    _RFLev = None

FUZZY_AVAILABLE = bool(_jellyfish and _RFLev)


def _phonetic_match(a: str, b: str) -> bool:
    if not _jellyfish:
        return False
    try:
        return (_jellyfish.metaphone(a) == _jellyfish.metaphone(b)
                or _jellyfish.soundex(a) == _jellyfish.soundex(b))
    except Exception:  # noqa: BLE001
        return False


def _lev_sim(a: str, b: str) -> float:
    if _RFLev:
        return _RFLev.normalized_similarity(a, b)
    # Minimal pure-python fallback so confidence gating still works without deps.
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def _preserve_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def fuzzy_correct_text(text: str, terms: list[str], threshold: float = 0.74) -> str:
    """Snap near-miss single words to known vocabulary. High-precision: only
    alpha tokens length>=3, length-ratio guarded, phonetic+edit-distance scored,
    case preserved. Surrounding spacing/punctuation is untouched."""
    if not text or not terms or not FUZZY_AVAILABLE:
        return text
    parts = re.split(r"(\W+)", text)   # keeps the delimiters in place
    for i, tok in enumerate(parts):
        if len(tok) < 3 or not tok.isalpha():
            continue
        low = tok.lower()
        best, best_score = None, 0.0
        for term in terms:
            tl = term.lower()
            if low == tl:
                best = None
                break  # already correct — never touch it
            m = max(len(tok), len(term))
            if m and abs(len(tok) - len(term)) / m > 0.34:
                continue
            score = _lev_sim(low, tl) + (0.25 if _phonetic_match(tok, term) else 0.0)
            if score > best_score:
                best_score, best = score, term
        if best and best_score >= threshold:
            parts[i] = _preserve_case(tok, best)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Automatic learning from your edits (no manual --correct, ever)
# ---------------------------------------------------------------------------
#
# This is the loop the user actually asked for: after we paste text, we keep a
# handle on the field we typed into. If you then fix a word, we diff your final
# text against what we pasted and — when the change is phonetically a real
# mis-hear of a distinctive (proper-noun-ish) word — we learn it silently. A
# phonetic gate is what separates "it typed Versal, I fixed it to Vercel" (learn)
# from "I changed my mind and rewrote the sentence" (ignore). Reads the field via
# the Accessibility API, which we already require for pasting — no new permission.

_NAME_RE = re.compile(r"[^\W\d_][\w'\-]*", re.UNICODE)


def _focused_app_name() -> str | None:
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return str(app.localizedName()) if app is not None else None
    except Exception:  # noqa: BLE001
        return None


def _ax_focused_element():
    """The AXUIElement with keyboard focus system-wide, or None. Needs the
    Accessibility permission (already required for paste)."""
    if not _ax_trusted():
        return None
    try:
        from ApplicationServices import (
            AXUIElementCreateSystemWide,
            AXUIElementCopyAttributeValue,
            kAXFocusedUIElementAttribute,
        )
        sysw = AXUIElementCreateSystemWide()
        err, el = AXUIElementCopyAttributeValue(
            sysw, kAXFocusedUIElementAttribute, None)
        return el if err == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _ax_read_value(el) -> str | None:
    """Current text value of an AX element (works for native NSText fields and
    many web/Electron fields; returns None when the app doesn't expose it)."""
    if el is None:
        return None
    try:
        from ApplicationServices import (
            AXUIElementCopyAttributeValue, kAXValueAttribute,
        )
        err, val = AXUIElementCopyAttributeValue(el, kAXValueAttribute, None)
        if err == 0 and isinstance(val, str):
            return val
    except Exception:  # noqa: BLE001
        return None
    return None


def learn_from_edit(el, pasted: str, *, max_learn: int = 3) -> int:
    """Diff the field's CURRENT text against what we pasted; auto-learn any
    distinctive word the user replaced with a phonetically-similar one. Returns
    the number of corrections learned."""
    current = _ax_read_value(el)
    if not current or not pasted:
        return 0
    pasted_words = _NAME_RE.findall(pasted)
    cur_words = _NAME_RE.findall(current)
    if not pasted_words or not cur_words:
        return 0
    pasted_set = {w.lower() for w in pasted_words}
    cur_set = {w.lower() for w in cur_words}
    # Sanity gate: only learn if MOST of what we pasted is still in the field —
    # otherwise the AX value is unrelated (user navigated away / different field)
    # and any "diff" would be noise.
    present = sum(1 for w in pasted_words if w.lower() in cur_set)
    if present < max(1, len(pasted_words) * 0.5):
        return 0
    # New words the user introduced (candidates for the corrected spelling).
    introduced = [w for w in cur_words if w.lower() not in pasted_set]
    learned = 0
    for w in pasted_words:
        if learned >= max_learn:
            break
        if len(w) < 3 or w.lower() in _COMMON_WORDS or w.lower() in cur_set:
            continue
        # w was distinctive AND is now gone — find its phonetic replacement.
        if not _is_distinctive(w):
            continue
        best, best_sim = None, 0.0
        for x in introduced:
            if len(x) < 2 or x.lower() == w.lower() or x.lower() in _COMMON_WORDS:
                continue
            sim = _lev_sim(w.lower(), x.lower())
            if (_phonetic_match(w, x) or sim >= 0.6) and sim > best_sim:
                best_sim, best = sim, x
        if best is not None:
            add_correction(w, best, silent=True)
            learned += 1
    if learned:
        print(f"[flow] auto-learned {learned} correction(s) from your edit ✓",
              flush=True)
    return learned


# ---------------------------------------------------------------------------
# Post-processing / cleanup
# ---------------------------------------------------------------------------

# Conservative: only strip true vocal fillers. Whisper already punctuates and
# capitalizes, so we don't try to rewrite meaning (that's what "llm" mode is for).
_FILLER_RE = re.compile(
    r"\b(?:u+h+|u+m+|a+h+|e+h+|e+r+m*|hm+|mm+-?hmm+|uh-huh)\b[,.]?",
    re.IGNORECASE,
)


def basic_cleanup(text: str) -> str:
    text = _FILLER_RE.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)   # no space before punctuation
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def llm_cleanup(text: str, cfg: dict, context: dict | None = None) -> str:
    """Polish dictation with a fast Anthropic model — FORMATTING ONLY, never
    rewriting your words. Needs ANTHROPIC_API_KEY.

    The prompt is built from the research consensus on why cloud tools feel more
    accurate than raw Whisper AND why they get the "the AI rewrote what I said"
    complaint: the cleanup layer must fix punctuation/casing/fillers but must NOT
    second-guess word identity (that's the recognizer's job). We also inject your
    known spellings (so it prefers them) and the active app (for tone)."""
    from anthropic import Anthropic
    context = context or {}
    client = Anthropic()

    blocks = []
    glossary = distinctive_terms(max_terms=60)
    if glossary:
        blocks.append("<known_spellings>\n" + ", ".join(glossary)
                      + "\n</known_spellings>")
    if context.get("app"):
        blocks.append(f"<active_app>{context['app']}</active_app>")
    ctx = ("\n\n" + "\n".join(blocks)) if blocks else ""

    system = (
        "You are a transcription editor for a voice-dictation tool. You receive a "
        "raw speech-to-text transcript and return it lightly cleaned up.\n"
        "HARD RULES:\n"
        "1. NEVER change the meaning, wording, or intent. You are not an assistant: "
        "do not answer questions, add content, or paraphrase.\n"
        "2. NEVER 'fix' a word you think was misheard by swapping in a different "
        "word. Word identity belongs to the speech recognizer, not you.\n"
        "3. DO fix punctuation, capitalization, and spacing; remove filler words "
        "(um, uh, like, you know) and false starts/stutters; honor an explicit "
        "spoken self-correction ('no wait, make that…', 'scratch that').\n"
        "4. If a word matches one of the user's known spellings below, prefer that "
        "exact spelling.\n"
        "5. Keep the tone the user dictated. Output ONLY the cleaned text — no "
        "preamble, no quotes, no commentary." + ctx
    )
    msg = client.messages.create(
        model=cfg.get("llm_model", "claude-haiku-4-5-20251001"),
        max_tokens=4096,   # headroom so long dictation isn't silently truncated
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def clean(text: str, cfg: dict, context: dict | None = None) -> str:
    if not text:
        return text
    mode = cfg["cleanup"]
    if mode == "none":
        out = text
    elif mode == "llm":
        try:
            out = llm_cleanup(text, cfg, context)
        except Exception as e:  # noqa: BLE001  fall back, never lose the words
            print(f"[flow] llm cleanup failed ({e}); using basic cleanup.")
            out = basic_cleanup(text)
    else:
        out = basic_cleanup(text)
    # Exact taught corrections first (precise), then phonetic fuzzy repair of any
    # remaining near-miss proper nouns against your learned vocabulary.
    out = apply_corrections(out)
    if cfg.get("fuzzy_correct", True):
        out = fuzzy_correct_text(out, distinctive_terms(),
                                 float(cfg.get("fuzzy_threshold", 0.74)))
    return out


# ---------------------------------------------------------------------------
# Text insertion (macOS)
# ---------------------------------------------------------------------------

def _pbpaste() -> str | None:
    try:
        return subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return None


def _pbcopy(text: str) -> None:
    subprocess.run(["pbcopy"], input=text, text=True)


# In-process pasteboard via AppKit (pyobjc-framework-Cocoa is already a dependency).
# This replaces the pbcopy/pbpaste subprocess round-trips on the hot paste path:
# synchronous, sub-millisecond, locale-independent, and it exposes changeCount() so we
# can tell whether anything else touched the clipboard without string-comparison polling.
_NSPB = None


def _pasteboard():
    global _NSPB
    if _NSPB is None:
        from AppKit import NSPasteboard
        _NSPB = NSPasteboard.generalPasteboard()
    return _NSPB


def _clip_get() -> str | None:
    """Current clipboard text in-process, or None if it isn't text (image, etc.)."""
    try:
        from AppKit import NSPasteboardTypeString
        return _pasteboard().stringForType_(NSPasteboardTypeString)
    except Exception:  # noqa: BLE001
        return _pbpaste()


def _clip_set(text: str) -> int:
    """Put `text` on the clipboard. Returns the pasteboard changeCount after the
    write (or -1 if AppKit was unavailable and we fell back to pbcopy)."""
    try:
        from AppKit import NSPasteboardTypeString
        pb = _pasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)
        return int(pb.changeCount())
    except Exception:  # noqa: BLE001
        _pbcopy(text)
        return -1


def _clip_change_count() -> int:
    try:
        return int(_pasteboard().changeCount())
    except Exception:  # noqa: BLE001
        return -1


def _ax_trusted() -> bool:
    """True iff this process is trusted for Accessibility (synthetic keystrokes).

    NOTE (macOS gotcha): AXIsProcessTrusted() is cached per-process at first
    call, so granting Accessibility to an already-running Terminal will NOT take
    effect until that Terminal is fully quit (Cmd-Q) and relaunched.
    """
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:  # noqa: BLE001  if we can't tell, assume trusted and try
        return True


def _secure_input_active() -> bool:
    """True iff macOS Secure Keyboard Entry is on (it silently blocks Cmd-V).

    Terminal has a "Secure Keyboard Entry" menu item that, when enabled, blocks
    ALL synthetic keystroke injection session-wide, even with Accessibility
    granted. Password fields trigger it transiently too.
    """
    try:
        import ctypes
        cg = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/"
            "ApplicationServices")
        cg.CGSIsSecureEventInputSet.restype = ctypes.c_bool
        return bool(cg.CGSIsSecureEventInputSet())
    except Exception:  # noqa: BLE001
        return False


def _send_paste() -> None:
    """Send Cmd-V as a single Quartz CGEvent pair.

    More reliable than pynput on macOS 15: the Command flag is set on the SAME
    event as the V key (no modifier race), and we set ONLY Command — so if the
    user is still physically holding Option, it can't turn into Cmd-Opt-V.
    Requires Accessibility, same as any synthetic keystroke.
    """
    import Quartz
    V_KEYCODE = 9  # 'v' on the US layout; layout-independent for Cmd-V chords.
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    down = Quartz.CGEventCreateKeyboardEvent(src, V_KEYCODE, True)
    up = Quartz.CGEventCreateKeyboardEvent(src, V_KEYCODE, False)
    Quartz.CGEventSetFlags(down, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventSetFlags(up, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)


def _send_backspaces(n: int) -> None:
    """Delete the `n` characters to the LEFT of the cursor by synthesizing that
    many Backspace (Delete) key presses as Quartz CGEvents — the same mechanism
    (and the same Accessibility requirement) as the synthetic Cmd-V paste. A tiny
    inter-event gap keeps fast apps (browsers/Electron) from coalescing/dropping
    events on long deletes."""
    if n <= 0:
        return
    import Quartz
    DELETE_KEYCODE = 51  # Backspace ("delete to the left")
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for _ in range(n):
        down = Quartz.CGEventCreateKeyboardEvent(src, DELETE_KEYCODE, True)
        up = Quartz.CGEventCreateKeyboardEvent(src, DELETE_KEYCODE, False)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
        time.sleep(0.003)


def _composed_len(s: str) -> int:
    """Number of Backspace presses needed to delete `s`: one per composed-character
    sequence, NOT per Python code point. A macOS Backspace removes a whole grapheme
    (e.g. NFD 'e'+combining-acute, or a flag emoji) at once, so counting code points
    (len) would send too many and chew into the user's other text. Falls back to len
    only if Foundation is somehow unavailable. For rare exotic emoji this may slightly
    UNDER-count (leaves a straggler) — the safe direction, since it never over-deletes."""
    try:
        from Foundation import NSString
        ns = NSString.stringWithString_(s)
        length = ns.length()
        i = n = 0
        while i < length:
            r = ns.rangeOfComposedCharacterSequenceAtIndex_(i)
            i = r.location + r.length
            n += 1
        return n
    except Exception:  # noqa: BLE001
        return len(s)


def _drop_last_sentence(s: str) -> str:
    """Return `s` with its last sentence removed (sentences end at . ! ?). Used so
    an inline 'actually never mind' retracts just the sentence spoken before it."""
    s = s.rstrip()
    if not s:
        return ""
    core = re.sub(r"[.!?]+$", "", s).rstrip()      # ignore a trailing terminator run
    terms = list(re.finditer(r"[.!?]+", core))
    return core[:terms[-1].end()] if terms else ""


def apply_undo(text: str, cfg: dict):
    """Apply 'never mind' retractions to ONE dictation, inline.

    Returns (kept_text, prev_delete_count):
      • Each undo phrase deletes the sentence spoken right before it *within this
        utterance* — so you can talk, say "actually never mind", and keep going in
        the same breath; only the retracted sentence is dropped from what's typed.
      • If a phrase has nothing before it in this utterance (it's the whole thing,
        or at the very start), it instead retracts a PREVIOUS pasted dictation
        (prev_delete_count) — the original separate-press behaviour still works.

    kept_text is what to actually type; prev_delete_count is how many earlier
    dictations to backspace away first. Returns (text, 0) when no phrase is present."""
    phrases = [str(p).strip() for p in (cfg.get("undo_phrases") or []) if str(p).strip()]
    if not text or not phrases:
        return text, 0
    pats = sorted({p.lower() for p in phrases}, key=len, reverse=True)  # longest first
    rx = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(p) for p in pats) + r")(?!\w)",
                    re.IGNORECASE)
    if not rx.search(text):
        return text, 0
    kept = ""
    prev_delete = 0
    pos = 0
    for m in rx.finditer(text):
        kept += text[pos:m.start()]
        if kept.strip():
            kept = _drop_last_sentence(kept)   # retract the sentence just spoken
        else:
            prev_delete += 1                   # nothing here yet -> retract a prior paste
            kept = ""
        pos = m.end()
    kept += text[pos:]
    # Tidy the seams left by the removals (stray/leading/duplicated punctuation).
    kept = re.sub(r"\s+", " ", kept)
    kept = re.sub(r"\s+([,.!?;:])", r"\1", kept)
    kept = re.sub(r"([.!?])\s*[.!?,;:]+", r"\1", kept)
    kept = re.sub(r"^[\s,;:.!?]+", "", kept)
    return kept.strip(), prev_delete


def insert_text(text: str, cfg: dict) -> bool:
    """Insert `text` into the focused app. Returns True if it was delivered
    via synthetic paste/type, False if it was only left on the clipboard as a
    fallback (so the caller can sound a distinct cue). The dictated text is
    NEVER lost: on any failure path it is left on the clipboard.
    """
    if not text:
        return False

    if cfg["insert_method"] == "clipboard":
        # Copy only — needs NO Accessibility permission. You press Cmd-V to
        # paste. The most permission-light way to get text out of Wispr DIY.
        _pbcopy(text)
        print("[flow] ✓ copied to clipboard — press Cmd-V to paste it.")
        return False

    if cfg["insert_method"] == "type":
        # Typed insert never touches the clipboard. Layout caveat applies, and
        # any '\n' will submit forms; paste is the safer default.
        if not _ax_trusted():
            _pbcopy(text)
            print("[flow] Accessibility NOT granted — left text on clipboard. "
                  "Grant Accessibility to Terminal, then Cmd-Q & relaunch.")
            return False
        from pynput.keyboard import Controller
        Controller().type(text)
        return True

    # ---- paste method (default) -------------------------------------------
    # Guard 1: Accessibility. If untrusted, Cmd-V is a silent no-op, so DON'T
    # paste and DON'T touch/restore the clipboard beyond leaving our text on it.
    if not _ax_trusted():
        _pbcopy(text)
        print("[flow] Accessibility NOT granted to this process — text left on "
              "clipboard, press Cmd-V. (Grant Accessibility to Terminal, then "
              "FULLY QUIT Terminal with Cmd-Q and relaunch — the grant is "
              "cached per-process and won't apply to a running Terminal.)")
        return False

    # Guard 2: Secure Input blocks synthetic Cmd-V even with Accessibility.
    if _secure_input_active():
        _pbcopy(text)
        print("[flow] Secure Keyboard Entry is ON — synthetic paste is blocked. "
              "Text left on clipboard; press Cmd-V. (Disable Terminal ▸ Secure "
              "Keyboard Entry, or click out of the password field.)")
        return False

    # Stash the existing clipboard (only if it's text we'd want back). Use a
    # truthy check, not `is not None`: _clip_get() returns None/"" for a non-text
    # clipboard (e.g. an image), and copying that back would wipe it.
    old = _clip_get() if cfg["restore_clipboard"] else None

    # In-process write is synchronous and confirmed on return — no poll loop needed
    # (the old pbcopy subprocess needed one; NSPasteboard does not).
    our_count = _clip_set(text)
    _send_paste()                       # Quartz Cmd-V — the text lands right here.

    # Restore the old clipboard AFTER the target app has consumed the paste, on a
    # background timer so this call returns IMMEDIATELY (the success ding, the
    # auto-learn arming, and readiness for the next dictation no longer wait ~0.25s).
    # Only restore if nothing else changed the pasteboard since our write, so a
    # failed paste still leaves the dictated text recoverable on the clipboard.
    if old and old != text:
        def _restore(expected=our_count, prev=old):
            if expected < 0 or _clip_change_count() == expected:
                _clip_set(prev)
        t = threading.Timer(0.4, _restore)
        t.daemon = True
        t.start()
    return True


# ---------------------------------------------------------------------------
# Feedback (subtle macOS sounds)
# ---------------------------------------------------------------------------

def play(sound: str, cfg: dict, volume: float = 1.0) -> None:
    if not cfg["play_sounds"]:
        return
    path = f"/System/Library/Sounds/{sound}.aiff"
    try:
        subprocess.Popen(
            ["afplay", "-v", str(volume), path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Hotkey handling + app loop
# ---------------------------------------------------------------------------
#
# IMPORTANT (macOS, diagnosed June 2026):
# We do NOT use pynput's high-level keyboard.Listener here. On this machine
# (macOS 15.7, Python 3.14 framework build) the launchd agent runs with
# `spawn type = daemon` and pynput taps at kCGHIDEventTap, which requires
# *effective* Accessibility (AXIsProcessTrusted). In the daemon context that
# trust is NOT effective: the tap is created (banner prints, no "not trusted"
# error) but never delivers a single event — exactly the silent dead-hotkey bug.
#
# A listen-only Quartz CGEventTap (kCGEventTapOptionListenOnly) needs only
# Input Monitoring (CGPreflightListenEventAccess), which IS granted here. So we
# build the tap directly via pyobjc/Quartz, match the Option keys by their
# virtual keycode (vk 58 = left Option, vk 61 = right Option), and re-enable the
# tap if macOS ever disables it. Everything downstream (record -> faster-whisper
# -> cleanup -> clipboard+Cmd-V paste) is unchanged.

# Virtual keycodes (kCGKeyCode) for the modifier keys we may use as a hotkey.
# These are stable macOS HID keycodes, independent of keyboard layout.
_VK_BY_NAME = {
    "alt_l": 58, "alt_r": 61,          # left / right Option
    "ctrl_l": 59, "ctrl_r": 62,        # left / right Control
    "cmd_l": 55, "cmd_r": 54,          # left / right Command
    "shift_l": 56, "shift_r": 60,      # left / right Shift
}
# Option keys are interchangeable for push-to-talk: if the configured hotkey is
# either Option, accept BOTH so layout/canonicalization quirks can't break it.
_OPTION_VKS = {58, 61}

# Modifier keys arrive as kCGEventFlagsChanged (a *bare* modifier produces NO
# keyDown/keyUp). For those we can't tell press from release by the event type,
# but we CAN read it deterministically from the event's flag bits: if the
# matching modifier's mask bit is set, the key is now down; if cleared, it's up.
# This is far more robust than toggling a Python-side flag, which gets stuck
# inverted forever if a single flagsChanged event is ever missed or doubled.
# (Mask values are filled in lazily from Quartz at run() time, since the
# constants live in the Quartz module.)
_MODIFIER_MASK_BY_VK: dict[int, int] = {}


def resolve_target_vks(name: str) -> set[int]:
    """Return the set of virtual keycodes that should trigger recording.

    Robust matching per the D1/D3/D4 diagnostics: never rely on a single key
    identity. If the user picked an Option key, accept either Option; otherwise
    accept the specific modifier's vk, or a single character's vk.
    """
    if name in ("alt_l", "alt_r"):
        return set(_OPTION_VKS)
    if name in _VK_BY_NAME:
        return {_VK_BY_NAME[name]}
    if len(name) == 1:
        # Resolve a single character to its vk via pynput's KeyCode table.
        from pynput.keyboard import KeyCode
        vk = getattr(KeyCode.from_char(name), "vk", None)
        if vk is not None:
            return {vk}
    raise ValueError(
        f"Unknown hotkey '{name}'. Use a modifier name (e.g. alt_r, cmd_r, "
        f"ctrl_r) or a single character."
    )


def _trust_probe() -> None:
    """Log the effective TCC state so flow.log is self-diagnosing, not silent."""
    try:
        from ApplicationServices import AXIsProcessTrusted
        print(f"[flow] AXIsProcessTrusted={AXIsProcessTrusted()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[flow] trust-probe (AX) unavailable: {e}", flush=True)
    try:
        from Quartz import CGPreflightListenEventAccess
        print(f"[flow] CGPreflightListenEventAccess="
              f"{CGPreflightListenEventAccess()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[flow] trust-probe (ListenEvent) unavailable: {e}", flush=True)


# ---------------------------------------------------------------------------
# Menu-bar ("real macOS app") support
#
# When launched as frutflow.app, flow.py runs an NSApplication in accessory mode
# so it owns a menu-bar status item (glyph = state, dropdown = Teach a Word /
# Restart / Open Log / Quit). None of this touches the dictation pipeline; it is
# purely additive and only active in app mode. All AppKit imports are deferred so
# the CLI paths (--correct, --try, …) never pay for them.
# ---------------------------------------------------------------------------

APP_BUNDLE_PATH = "/Users/thorstenpfeiffer/Applications/frutflow.app"
APP_BUNDLE_ID = "com.frutflow.dictation"
AGENT_LABEL = "com.wisprdiy.dictation"


def _teach_word_interactive() -> None:
    """Menu 'Teach a Word…' — the same two-dialog flow as the .command file, but
    in-process. Run on a background thread so the menu click never blocks the run
    loop while the dialogs are up."""
    def _ask(prompt: str, save: bool = False) -> str:
        btn = "Save" if save else "Next"
        # ensure_ascii=False is REQUIRED: AppleScript string literals accept
        # literal UTF-8 but reject json's \\uXXXX escapes, so accented words
        # (früt, café) would otherwise fail to compile and silently not save.
        script = (
            'try\n'
            f'  set r to text returned of (display dialog {json.dumps(prompt, ensure_ascii=False)} '
            f'default answer "" with title "Teach Wispr DIY" '
            f'buttons {{"Cancel", "{btn}"}} default button "{btn}")\n'
            '  return r\n'
            'on error\n  return ""\nend try'
        )
        try:
            out = subprocess.run(["osascript", "-e", script],
                                 capture_output=True, text=True, timeout=300)
            return out.stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    heard = _ask("Teach Wispr DIY a fix.\n\nWhat did it type WRONG?  "
                 "(the word it got wrong)")
    if not heard:
        return
    correct = _ask(f'What should it have typed instead of "{heard}"?', save=True)
    if not correct:
        return
    try:
        add_correction(heard, correct)
    except Exception as e:  # noqa: BLE001
        print(f"[flow] teach-a-word failed: {e}", flush=True)
        return
    msg = (f'Saved. Wispr DIY will now type "{correct}" instead of "{heard}" '
           'from your next dictation on.')
    try:
        subprocess.run(["osascript", "-e",
                        f'display dialog {json.dumps(msg, ensure_ascii=False)} '
                        'with title "Wispr DIY" '
                        'buttons {"Great"} default button "Great"'],
                       capture_output=True, text=True, timeout=120)
    except Exception:  # noqa: BLE001
        pass


_MENU_ACTIONS_CLASS = None


def _menu_actions_class():
    """Lazily build (and cache) the NSObject subclass that backs the menu-bar
    item's actions. Deferred import so non-app code paths never touch AppKit."""
    global _MENU_ACTIONS_CLASS
    if _MENU_ACTIONS_CLASS is not None:
        return _MENU_ACTIONS_CLASS
    import objc
    from Cocoa import NSObject, NSApplication

    class _MenuActions(NSObject):
        def initWithApp_(self, app):
            self = objc.super(_MenuActions, self).init()
            if self is None:
                return None
            self._app = app
            return self

        # Called on the main thread (via performSelectorOnMainThread) to reflect
        # the current dictation state in the menu bar. Never raises.
        def applyStatus_(self, arr):
            try:
                app = self._app
                glyph, label = arr[0], arr[1]
                item = app._status_item
                if item is not None:
                    btn = item.button()
                    if btn is not None:
                        btn.setTitle_(glyph)
                    else:
                        item.setTitle_(glyph)
                if app._status_line is not None:
                    app._status_line.setTitle_(label)
                # Same state, second surface: the floating recording HUD.
                app._hud_apply(glyph)
            except Exception:  # noqa: BLE001
                pass

        def teachWord_(self, sender):
            threading.Thread(target=_teach_word_interactive, daemon=True).start()

        def openLog_(self, sender):
            try:
                subprocess.Popen(["open",
                                  str(Path.home() / ".flowdictate" / "flow.log")])
            except Exception:  # noqa: BLE001
                pass

        def openPrivacy_(self, sender):
            # Jump straight to the Input Monitoring pane (the grant needed for the
            # hotkey tap). Accessibility + Microphone live one click away in the
            # same Privacy & Security list.
            try:
                subprocess.Popen(["open",
                    "x-apple.systempreferences:com.apple.preference.security"
                    "?Privacy_ListenEvent"])
            except Exception:  # noqa: BLE001
                pass

        def transcribeFile_(self, sender):
            try:
                self._app._show_transcribe_window()
            except Exception:  # noqa: BLE001
                pass

        def historyWindow_(self, sender):
            try:
                self._app._show_history_window()
            except Exception:  # noqa: BLE001
                pass

        # NSApplication delegate: fires when you double-click the app (or click its
        # Dock icon) while it's ALREADY running. A menu-bar app has no main window,
        # so without this "opening" the app does nothing visible — here we open the
        # History window (the home page) so it behaves like a normal app you open.
        def applicationShouldHandleReopen_hasVisibleWindows_(self, app, flag):
            try:
                self._app._show_history_window()
            except Exception:  # noqa: BLE001
                pass
            return True

        def restart_(self, sender):
            # Relaunch a fresh instance, then quit this one. Detached so it
            # survives our termination; LSMultipleInstancesProhibited + our exit
            # ensure exactly one instance ends up running.
            try:
                subprocess.Popen(
                    ["/bin/sh", "-c", f'sleep 1; open -g "{APP_BUNDLE_PATH}"'],
                    start_new_session=True)
            except Exception:  # noqa: BLE001
                pass
            NSApplication.sharedApplication().terminate_(None)

        def quit_(self, sender):
            # Full stop matching "Quit frutflow.command": disable the self-heal /
            # login agent FIRST so the watchdog can't relaunch us, then terminate.
            # (frutflow starts again at next login, or when you reopen the app.)
            try:
                subprocess.run(
                    ["launchctl", "bootout", f"gui/{os.getuid()}/{AGENT_LABEL}"],
                    capture_output=True)
            except Exception:  # noqa: BLE001
                pass
            NSApplication.sharedApplication().terminate_(None)

    _MENU_ACTIONS_CLASS = _MenuActions
    return _MENU_ACTIONS_CLASS


# ---------------------------------------------------------------------------
# Liquid-glass UI helpers, shared by the History + Transcribe windows.
# All AppKit imports are deferred so CLI paths never load Cocoa. Every newer API
# (continuous corner curve, SF-Rounded font design, some materials) is
# version-guarded: an older macOS degrades to a plain-but-fine look, not a crash.
# ---------------------------------------------------------------------------
_GLASS = None


def _glass():
    """Lazily build & cache the glass helper namespace (a class with static
    methods + resolved constants)."""
    global _GLASS
    if _GLASS is not None:
        return _GLASS

    from Cocoa import (
        NSView, NSVisualEffectView, NSColor, NSFont,
        NSMakeRect, NSMakeSize,
        NSViewWidthSizable, NSViewHeightSizable,
    )

    # Vibrancy material / blend / state — import by name, fall back to the stable
    # raw enum values (an unimported constant is a runtime NameError, not compile).
    try:
        from Cocoa import (
            NSVisualEffectMaterialUnderWindowBackground as _MAT_WINDOW,
            NSVisualEffectMaterialSidebar as _MAT_SIDEBAR,
            NSVisualEffectMaterialPopover as _MAT_CARD,
            NSVisualEffectMaterialHeaderView as _MAT_HEADER,
            NSVisualEffectBlendingModeBehindWindow as _BLEND_BEHIND,
            NSVisualEffectBlendingModeWithinWindow as _BLEND_WITHIN,
            NSVisualEffectStateActive as _STATE_ACTIVE,
        )
    except ImportError:  # pragma: no cover — very old pyobjc
        _MAT_WINDOW, _MAT_SIDEBAR, _MAT_CARD, _MAT_HEADER = 21, 7, 6, 10
        _BLEND_BEHIND, _BLEND_WITHIN, _STATE_ACTIVE = 0, 1, 1

    # SF-Rounded design token (macOS 10.15+).
    try:
        from Cocoa import NSFontDescriptorSystemDesignRounded as _ROUNDED
    except ImportError:  # pragma: no cover
        _ROUNDED = None

    # Continuous "squircle" corner curve. On this pyobjc it lives in Quartz; the
    # KVC string value 'continuous' is an equivalent fallback.
    _CURVE = None
    try:
        from Quartz import kCACornerCurveContinuous as _CURVE
    except Exception:  # noqa: BLE001
        _CURVE = "continuous"

    _SIZABLE = NSViewWidthSizable | NSViewHeightSizable
    _rounded_cache = {}

    class _Glass:
        MAT_WINDOW, MAT_SIDEBAR = _MAT_WINDOW, _MAT_SIDEBAR
        MAT_CARD, MAT_HEADER = _MAT_CARD, _MAT_HEADER
        BLEND_BEHIND, BLEND_WITHIN, STATE_ACTIVE = _BLEND_BEHIND, _BLEND_WITHIN, _STATE_ACTIVE

        @staticmethod
        def rounded_font(size, weight=0.0):
            """SF Rounded at size/weight (the iOS vibe); falls back to the plain
            system font where the rounded design is unavailable. Cached."""
            key = (round(float(size), 1), float(weight))
            f = _rounded_cache.get(key)
            if f is not None:
                return f
            base = NSFont.systemFontOfSize_weight_(size, weight)
            out = base
            if _ROUNDED is not None:
                try:
                    desc = base.fontDescriptor().fontDescriptorWithDesign_(_ROUNDED)
                    if desc is not None:
                        r = NSFont.fontWithDescriptor_size_(desc, size)
                        if r is not None:
                            out = r
                except Exception:  # noqa: BLE001
                    out = base
            _rounded_cache[key] = out
            return out

        @staticmethod
        def dress_window(win):
            """Translucent edge-to-edge chrome so the window blur runs under the
            traffic lights. Safe on all recent macOS; degrades to a plain window."""
            try:
                from Cocoa import (NSWindowStyleMaskFullSizeContentView,
                                   NSWindowTitleHidden)
                win.setStyleMask_(win.styleMask() | NSWindowStyleMaskFullSizeContentView)
                win.setTitlebarAppearsTransparent_(True)
                win.setTitleVisibility_(NSWindowTitleHidden)
                win.setMovableByWindowBackground_(True)
            except Exception:  # noqa: BLE001
                pass

        @staticmethod
        def backing(frame, material):
            """A behind-window blur view to use as a window's content view."""
            v = NSVisualEffectView.alloc().initWithFrame_(frame)
            v.setBlendingMode_(_Glass.BLEND_BEHIND)
            v.setState_(_Glass.STATE_ACTIVE)
            try:
                v.setMaterial_(material)
            except Exception:  # noqa: BLE001
                pass
            v.setAutoresizingMask_(_SIZABLE)
            return v

        @staticmethod
        def round_layer(view, radius, mask=True):
            """Layer-back `view`, round it, and use the continuous curve when
            available. Caller owns any shadow (a masked layer can't cast one)."""
            view.setWantsLayer_(True)
            layer = view.layer()
            if layer is None:
                return None
            layer.setCornerRadius_(radius)
            layer.setMasksToBounds_(mask)
            if _CURVE is not None:
                try:
                    layer.setCornerCurve_(_CURVE)
                except Exception:  # noqa: BLE001
                    pass
            return layer

        @staticmethod
        def card(frame, radius=15.0):
            """A rounded translucent squircle card. Because a masksToBounds layer
            can't ALSO cast an outer shadow, return (container, inner):
              • container — unmasked NSView that carries the soft drop shadow,
              • inner     — within-window NSVisualEffectView clipped to a
                            continuous-rounded squircle with a hairline rim.
            Add content to `inner`."""
            container = NSView.alloc().initWithFrame_(frame)
            container.setWantsLayer_(True)
            cl = container.layer()
            if cl is not None:
                cl.setShadowColor_(NSColor.blackColor().CGColor())
                cl.setShadowOpacity_(0.16)
                cl.setShadowRadius_(9.0)
                cl.setShadowOffset_(NSMakeSize(0.0, -2.0))  # CA: -y = downward on screen
                cl.setMasksToBounds_(False)                 # MUST be false to cast a shadow

            inner = NSVisualEffectView.alloc().initWithFrame_(
                NSMakeRect(0, 0, frame.size.width, frame.size.height))
            inner.setBlendingMode_(_Glass.BLEND_WITHIN)
            inner.setState_(_Glass.STATE_ACTIVE)
            try:
                inner.setMaterial_(_Glass.MAT_CARD)
            except Exception:  # noqa: BLE001
                pass
            inner.setAutoresizingMask_(_SIZABLE)
            il = _Glass.round_layer(inner, radius, mask=True)
            if il is not None:
                il.setBorderWidth_(1.0)
                il.setBorderColor_(
                    NSColor.whiteColor().colorWithAlphaComponent_(0.14).CGColor())
            container.addSubview_(inner)
            return container, inner

    _GLASS = _Glass
    return _GLASS


def _sync_activation_policy():
    """Keep the Dock icon visible while ANY real app window is open; revert to
    menu-bar-only (Accessory) once the last one closes. Call (deferred one runloop
    turn) from every window's windowWillClose_. Main-thread only."""
    try:
        from Cocoa import (
            NSApplication, NSApplicationActivationPolicyRegular,
            NSApplicationActivationPolicyAccessory, NSWindowStyleMaskTitled,
        )
        app = NSApplication.sharedApplication()
        any_visible = False
        for w in app.windows():
            try:
                # Only count real, on-screen, titled windows (the status item is
                # not a window and won't appear here; skip off-screen panels).
                if (w.isVisible() and not w.isMiniaturized()
                        and (w.styleMask() & NSWindowStyleMaskTitled)):
                    any_visible = True
                    break
            except Exception:  # noqa: BLE001
                pass
        app.setActivationPolicy_(
            NSApplicationActivationPolicyRegular if any_visible
            else NSApplicationActivationPolicyAccessory)
    except Exception:  # noqa: BLE001
        pass


_TRANSCRIBE_CTRL_CLASS = None


def _transcribe_controller_class():
    """Lazily build the controller that runs the 'Transcribe an audio file' window:
    a titled, resizable window with a Choose-file button, a status line, an editable
    transcript text view, and Copy / Save buttons. Deferred import so CLI paths never
    load AppKit."""
    global _TRANSCRIBE_CTRL_CLASS
    if _TRANSCRIBE_CTRL_CLASS is not None:
        return _TRANSCRIBE_CTRL_CLASS
    import objc
    from Cocoa import (
        NSObject, NSWindow, NSScrollView, NSTextView, NSButton, NSTextField,
        NSOpenPanel, NSSavePanel, NSApplication, NSColor, NSFont,
        NSApplicationActivationPolicyRegular, NSApplicationActivationPolicyAccessory,
        NSMakeRect, NSMakeSize, NSOperationQueue,
        NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
        NSWindowStyleMaskResizable, NSWindowStyleMaskMiniaturizable,
        NSBackingStoreBuffered, NSViewWidthSizable, NSViewHeightSizable,
        NSViewMinYMargin, NSViewMaxYMargin, NSVisualEffectView,
    )
    G = _glass()
    OK = 1               # NSModalResponseOK / NSFileHandlingPanelOKButton
    AUDIO_TYPES = ["wav", "aiff", "aif", "aifc", "caf", "m4a", "m4b", "mp3", "mp4",
                   "aac", "flac", "ogg", "opus", "mov", "wma", "amr", "3gp"]

    class _TranscribeController(NSObject):
        def initWithApp_(self, app):
            self = objc.super(_TranscribeController, self).init()
            if self is None:
                return None
            self._app = app
            self._path = None
            self._busy = False
            self._build()
            return self

        @objc.python_method
        def _build(self):
            style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                     | NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable)
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 640, 520), style, NSBackingStoreBuffered, False)
            win.setTitle_("Transcribe Audio File — früt Flow")
            win.setReleasedWhenClosed_(False)
            win.setDelegate_(self)
            win.setMinSize_(NSMakeSize(440, 340))
            # Liquid-glass chrome: edge-to-edge blur under the traffic lights.
            G.dress_window(win)
            content = G.backing(win.contentView().frame(), G.MAT_WINDOW)
            win.setContentView_(content)

            choose = NSButton.buttonWithTitle_target_action_(
                "Choose Audio File…", self, "chooseFile:")
            choose.setFrame_(NSMakeRect(16, 476, 190, 28))
            choose.setAutoresizingMask_(NSViewMinYMargin)
            choose.setFont_(G.rounded_font(13))
            content.addSubview_(choose)
            self._choose = choose

            status = NSTextField.labelWithString_(
                "Choose an audio file (voice memo, m4a, mp3, wav…) to transcribe it.")
            status.setFrame_(NSMakeRect(216, 481, 408, 20))
            status.setAutoresizingMask_(NSViewMinYMargin | NSViewWidthSizable)
            status.setFont_(G.rounded_font(13))
            status.setTextColor_(NSColor.secondaryLabelColor())
            content.addSubview_(status)
            self._status = status

            scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(16, 52, 608, 412))
            scroll.setHasVerticalScroller_(True)
            scroll.setBorderType_(0)              # NSNoBorder (was NSBezelBorder)
            scroll.setDrawsBackground_(False)     # let the window blur show through
            scroll.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            G.round_layer(scroll, 12.0)           # continuous-rounded transcript panel
            tv = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 606, 410))
            tv.setEditable_(True)
            tv.setRichText_(False)
            tv.setFont_(G.rounded_font(14))
            tv.setDrawsBackground_(False)         # transparent over the blur
            tv.setTextColor_(NSColor.labelColor())
            tv.setTextContainerInset_(NSMakeSize(10, 10))
            tv.setAutoresizingMask_(NSViewWidthSizable)
            scroll.setDocumentView_(tv)
            content.addSubview_(scroll)
            self._tv = tv

            copy = NSButton.buttonWithTitle_target_action_("Copy", self, "copyText:")
            copy.setFrame_(NSMakeRect(16, 12, 96, 30))
            copy.setAutoresizingMask_(NSViewMaxYMargin)
            copy.setFont_(G.rounded_font(13))
            content.addSubview_(copy)
            save = NSButton.buttonWithTitle_target_action_("Save…", self, "saveText:")
            save.setFrame_(NSMakeRect(118, 12, 96, 30))
            save.setAutoresizingMask_(NSViewMaxYMargin)
            save.setFont_(G.rounded_font(13))
            content.addSubview_(save)

            win.center()
            self._win = win

        # -- helpers (pure-Python; hidden from the Obj-C runtime) -----------
        @objc.python_method
        def _set_status(self, s):
            self._status.setStringValue_(s)

        @objc.python_method
        def show(self):
            # Become a regular app while the window is open so it focuses and shows a
            # Dock icon (the app logo); revert to menu-bar-only when it closes.
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            app.activateIgnoringOtherApps_(True)
            self._win.makeKeyAndOrderFront_(None)

        def windowWillClose_(self, note):
            # Defer one runloop turn so the closing window is no longer "visible";
            # revert to Accessory only if no other app window remains (don't yank
            # the Dock icon while the History window is still open).
            NSOperationQueue.mainQueue().addOperationWithBlock_(
                lambda: _sync_activation_policy())

        # -- actions --------------------------------------------------------
        def chooseFile_(self, sender):
            if self._busy:
                return
            panel = NSOpenPanel.openPanel()
            panel.setCanChooseFiles_(True)
            panel.setCanChooseDirectories_(False)
            panel.setAllowsMultipleSelection_(False)
            panel.setAllowedFileTypes_(AUDIO_TYPES)
            panel.setMessage_("Choose an audio file to transcribe")
            if panel.runModal() != OK or not panel.URLs():
                return
            path = str(panel.URLs()[0].path())
            self._path = path
            name = os.path.basename(path)
            self._busy = True
            self._choose.setEnabled_(False)
            self._set_status(f"Transcribing “{name}” — long files can take a bit…")
            self._tv.setString_("")
            threading.Thread(target=self._run, args=(path, name), daemon=True).start()

        @objc.python_method
        def _run(self, path, name):
            try:
                text, err = self._app._transcribe_path(path)
            except Exception as e:  # noqa: BLE001
                text, err = None, f"Transcription failed: {e}"

            def _done():
                self._busy = False
                self._choose.setEnabled_(True)
                if err:
                    self._set_status(err)
                elif not text:
                    self._set_status("No speech detected in that file.")
                else:
                    self._set_status(f"Done — {len(text.split())} words from “{name}”. "
                                  "Edit, Copy, or Save below.")
                    self._tv.setString_(text)
            NSOperationQueue.mainQueue().addOperationWithBlock_(_done)

        def copyText_(self, sender):
            s = self._tv.string()
            if s and str(s).strip():
                _clip_set(str(s))
                self._set_status("Copied to the clipboard.")

        def saveText_(self, sender):
            s = self._tv.string()
            if not (s and str(s).strip()):
                return
            panel = NSSavePanel.savePanel()
            panel.setAllowedFileTypes_(["txt"])
            base = "transcription"
            if self._path:
                base = os.path.splitext(os.path.basename(self._path))[0] + " — transcript"
            panel.setNameFieldStringValue_(base + ".txt")
            if panel.runModal() == OK and panel.URL():
                try:
                    with open(str(panel.URL().path()), "w", encoding="utf-8") as f:
                        f.write(str(s))
                    self._set_status("Saved.")
                except Exception as e:  # noqa: BLE001
                    self._set_status(f"Couldn't save: {e}")

    _TRANSCRIBE_CTRL_CLASS = _TranscribeController
    return _TRANSCRIBE_CTRL_CLASS


_HISTORY_CTRL_CLASS = None


def _history_controller_class():
    """Lazily build the History window controller: an iOS 'liquid glass' list of
    the last 100 dictations (newest first) as translucent squircle cards in a
    flipped NSStackView, with a search field, per-card Copy, a Clear control, and
    an empty state. Re-reads history.json on every show(). Deferred AppKit import.
    Mirrors _transcribe_controller_class's patterns."""
    global _HISTORY_CTRL_CLASS
    if _HISTORY_CTRL_CLASS is not None:
        return _HISTORY_CTRL_CLASS
    import objc
    from Cocoa import (
        NSObject, NSView, NSWindow, NSScrollView, NSStackView, NSTextField,
        NSButton, NSImage, NSSearchField, NSAlert, NSApplication, NSColor,
        NSMakeRect, NSMakeSize, NSMakePoint, NSOperationQueue, NSTimer,
        NSApplicationActivationPolicyRegular, NSApplicationActivationPolicyAccessory,
        NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
        NSWindowStyleMaskResizable, NSWindowStyleMaskMiniaturizable,
        NSBackingStoreBuffered, NSViewWidthSizable, NSViewHeightSizable,
        NSViewMinXMargin, NSViewMinYMargin, NSViewMaxYMargin,
        NSTextAlignmentCenter, NSImageLeft,
    )
    # Layout constants that some pyobjc builds don't export by name (raw values
    # are stable across macOS): stack orientation / distribution / alignment.
    try:
        from Cocoa import (
            NSUserInterfaceLayoutOrientationVertical as _VERT,
            NSStackViewDistributionFill as _FILL,
            NSLayoutAttributeLeading as _ALIGN_LEADING,
        )
    except ImportError:  # pragma: no cover
        _VERT, _FILL, _ALIGN_LEADING = 1, 0, 5

    G = _glass()
    PAD = 16.0          # window inner padding
    CARD_GAP = 10.0     # vertical gap between cards
    TOPBAR_H = 70.0     # search + buttons row (leaves the top strip for traffic lights)

    # A flipped document view so the stack lays out TOP-DOWN (AppKit's default
    # origin is bottom-left); the newest card ends up at the top like an iOS list.
    class _FlippedDoc(NSView):
        def isFlipped(self):
            return True

    class _HistoryController(NSObject):
        def initWithApp_(self, app):
            self = objc.super(_HistoryController, self).init()
            if self is None:
                return None
            self._app = app
            self._all = []        # history dicts, newest-first, cached per show()
            self._rows = {}       # button tag -> row text (keeps Copy targets valid)
            self._bodies = []     # body labels, for responsive re-wrap on resize
            self._next_tag = 1
            self._filter_timer = None   # debounce live search (coalesce keystrokes)
            self._resize_timer = None   # coalesce live-drag resize re-layout
            self._build()
            return self

        # ---- window construction -----------------------------------------
        @objc.python_method
        def _build(self):
            style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                     | NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable)
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 560, 640), style, NSBackingStoreBuffered, False)
            win.setTitle_("History — früt Flow")
            win.setReleasedWhenClosed_(False)
            win.setDelegate_(self)
            win.setMinSize_(NSMakeSize(420, 380))
            G.dress_window(win)

            frame = win.contentView().frame()
            content = G.backing(frame, G.MAT_SIDEBAR)   # behind-window blur = content view
            win.setContentView_(content)
            W = frame.size.width
            H = frame.size.height
            ctrl_y = H - 62      # control row sits just below the traffic-light strip

            # --- top bar: search field + Transcribe + Clear -------------------
            search = NSSearchField.alloc().initWithFrame_(
                NSMakeRect(PAD, ctrl_y, W - PAD * 2 - 216, 30))
            search.setAutoresizingMask_(NSViewWidthSizable | NSViewMinYMargin)
            search.setFont_(G.rounded_font(13))
            search.setPlaceholderString_("Search dictations")
            search.setDelegate_(self)                 # controlTextDidChange_ -> live filter
            search.setTarget_(self)
            search.setAction_("searchChanged:")
            content.addSubview_(search)
            self._search = search

            trans = NSButton.buttonWithTitle_target_action_(
                "Transcribe File…", self, "openTranscribe:")
            trans.setFrame_(NSMakeRect(W - PAD - 208, ctrl_y, 122, 30))
            trans.setAutoresizingMask_(NSViewMinXMargin | NSViewMinYMargin)
            trans.setBezelStyle_(1)   # NSBezelStyleRounded
            trans.setFont_(G.rounded_font(13))
            content.addSubview_(trans)

            clear = NSButton.buttonWithTitle_target_action_(
                "Clear", self, "clearHistory:")
            clear.setFrame_(NSMakeRect(W - PAD - 78, ctrl_y, 78, 30))
            clear.setAutoresizingMask_(NSViewMinXMargin | NSViewMinYMargin)
            clear.setBezelStyle_(1)
            clear.setFont_(G.rounded_font(13))
            content.addSubview_(clear)
            self._clear = clear

            # --- scroll view + flipped stack of cards -------------------------
            scroll = NSScrollView.alloc().initWithFrame_(
                NSMakeRect(0, 0, W, H - TOPBAR_H))
            scroll.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            scroll.setHasVerticalScroller_(True)
            scroll.setDrawsBackground_(False)   # let the blur show through
            scroll.setBorderType_(0)            # NSNoBorder

            doc = _FlippedDoc.alloc().initWithFrame_(NSMakeRect(0, 0, W, 10))
            doc.setAutoresizingMask_(NSViewWidthSizable)

            stack = NSStackView.alloc().initWithFrame_(NSMakeRect(0, 0, W, 10))
            stack.setOrientation_(_VERT)
            stack.setAlignment_(_ALIGN_LEADING)
            stack.setDistribution_(_FILL)
            stack.setSpacing_(CARD_GAP)
            stack.setEdgeInsets_((PAD, PAD, PAD, PAD))   # top,left,bottom,right
            stack.setTranslatesAutoresizingMaskIntoConstraints_(False)
            doc.addSubview_(stack)
            # Pin the stack to the flipped doc: full width, top-anchored. Its height
            # is driven by the arranged cards (each sizes to its wrapped text).
            stack.leadingAnchor().constraintEqualToAnchor_(doc.leadingAnchor()).setActive_(True)
            stack.trailingAnchor().constraintEqualToAnchor_(doc.trailingAnchor()).setActive_(True)
            stack.topAnchor().constraintEqualToAnchor_(doc.topAnchor()).setActive_(True)

            scroll.setDocumentView_(doc)
            content.addSubview_(scroll)
            self._scroll = scroll
            self._doc = doc
            self._stack = stack

            # Empty-state label, centered; shown only when there are no cards.
            empty = NSTextField.labelWithString_("No dictations yet…")
            empty.setFont_(G.rounded_font(17, 0.2))
            empty.setTextColor_(NSColor.tertiaryLabelColor())
            empty.setAlignment_(NSTextAlignmentCenter)
            empty.setFrame_(NSMakeRect(0, (H - TOPBAR_H) / 2 - 16, W, 32))
            empty.setAutoresizingMask_(
                NSViewWidthSizable | NSViewMinYMargin | NSViewMaxYMargin)
            empty.setHidden_(True)
            content.addSubview_(empty)
            self._empty = empty

            win.center()
            self._win = win

        # ---- show / activation policy ------------------------------------
        @objc.python_method
        def show(self):
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            app.activateIgnoringOtherApps_(True)
            self._reload()                       # re-read history.json every show
            self._win.makeKeyAndOrderFront_(None)
            self._win.makeFirstResponder_(self._search)

        def windowWillClose_(self, note):
            # Kill any pending debounced work so a timer can't fire into a closed
            # window (touching a torn-down view hierarchy).
            self._cancel_timers()
            # Defer one runloop turn so the closing window is no longer counted as
            # visible, then revert to Accessory ONLY if no other window remains.
            NSOperationQueue.mainQueue().addOperationWithBlock_(
                lambda: _sync_activation_policy())

        def windowDidResize_(self, note):
            # Coalesce live-drag resize ticks: re-wrapping ~100 blur cards on every
            # intermediate frame is janky, so run the re-fit once the drag settles.
            if self._resize_timer is not None:
                self._resize_timer.invalidate()
            self._resize_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                0.05, False, lambda _t: self._resize_doc())

        @objc.python_method
        def _cancel_timers(self):
            for attr in ("_filter_timer", "_resize_timer"):
                t = getattr(self, attr, None)
                if t is not None:
                    try:
                        t.invalidate()
                    except Exception:  # noqa: BLE001
                        pass
                    setattr(self, attr, None)

        # ---- data -> cards -----------------------------------------------
        @objc.python_method
        def _reload(self):
            self._all = load_history()           # newest-first, best-effort
            self._rebuild(str(self._search.stringValue() or ""))

        @objc.python_method
        def _rebuild(self, query):
            """Tear down and repopulate the stack from self._all, filtered by
            query. Rebuilding the whole stack is the simplest correct filter; at
            the 100-cap it is imperceptible."""
            for v in list(self._stack.arrangedSubviews()):
                self._stack.removeArrangedSubview_(v)
                v.removeFromSuperview()
            self._rows.clear()
            self._bodies = []
            self._next_tag = 1

            q = query.strip().lower()
            items = self._all
            if q:
                items = [e for e in items
                         if q in str(e.get("text", "")).lower()
                         or q in str(e.get("app") or "").lower()]

            if not items:
                self._empty.setStringValue_("No matches." if q else "No dictations yet…")
                self._empty.setHidden_(False)
                avail = self._scroll.contentSize()
                self._doc.setFrameSize_(NSMakeSize(avail.width, avail.height))
                return
            self._empty.setHidden_(True)

            for e in items:
                card = self._make_card(e)
                self._stack.addArrangedSubview_(card)
                # Now that the card shares the stack's hierarchy, pin its width to
                # the stack's inset content width (full-width cards, PAD each side).
                card.widthAnchor().constraintEqualToAnchor_constant_(
                    self._stack.widthAnchor(), -2 * PAD).setActive_(True)
            self._resize_doc()
            # A reload or a filter change should always show results from the top,
            # not leave the flipped list parked in blank space where it was scrolled.
            self._doc.scrollPoint_(NSMakePoint(0, 0))

        @objc.python_method
        def _resize_doc(self):
            """Let Auto Layout size the stack, then match the flipped doc height to
            it so the scroller reflects real content height. Re-wraps body labels
            to the current width first (so they report a correct multi-line height)."""
            self._stack.layoutSubtreeIfNeeded()
            for b in self._bodies:
                w = b.frame().size.width
                if w > 1:
                    b.setPreferredMaxLayoutWidth_(w)
            self._stack.layoutSubtreeIfNeeded()
            h = self._stack.fittingSize().height
            avail = self._scroll.contentSize()
            self._doc.setFrameSize_(NSMakeSize(avail.width, max(h, avail.height)))

        @objc.python_method
        def _make_card(self, entry):
            """One translucent glass card: wrapping+selectable body text, a subtle
            meta line, and a Copy button. Auto Layout sizes the card to its text.
            The Copy button targets `self` (the retained controller); the row text
            is stashed in self._rows[tag] — so there is NO per-card objc object
            that could be GC'd out from under the run loop."""
            text = str(entry.get("text", ""))
            app_name = entry.get("app")
            words = int(entry.get("words") or len(text.split()))
            when = relative_time(entry.get("ts"))
            delivered = bool(entry.get("delivered", True))

            container, inner = G.card(NSMakeRect(0, 0, 480, 60), radius=15.0)
            container.setTranslatesAutoresizingMaskIntoConstraints_(False)
            inner.setTranslatesAutoresizingMaskIntoConstraints_(False)
            # (The card's width is pinned to the stack AFTER it's added as an
            # arranged subview — see _rebuild — so the anchors share an ancestor.)
            # Inner fills the container (container is the arranged subview).
            inner.leadingAnchor().constraintEqualToAnchor_(container.leadingAnchor()).setActive_(True)
            inner.trailingAnchor().constraintEqualToAnchor_(container.trailingAnchor()).setActive_(True)
            inner.topAnchor().constraintEqualToAnchor_(container.topAnchor()).setActive_(True)
            inner.bottomAnchor().constraintEqualToAnchor_(container.bottomAnchor()).setActive_(True)

            # Body: wrapping, selectable dictation text. Both horizontal edges are
            # pinned, so the label wraps to that width and self-sizes its height.
            body = NSTextField.wrappingLabelWithString_(text)
            body.setSelectable_(True)
            body.setFont_(G.rounded_font(14))
            body.setTextColor_(NSColor.labelColor())
            body.setTranslatesAutoresizingMaskIntoConstraints_(False)
            inner.addSubview_(body)
            self._bodies.append(body)

            # Meta line: "2m ago  ·  7 words  ·  Safari" (+ clipboard note).
            bits = []
            if when:
                bits.append(when)
            bits.append("1 word" if words == 1 else f"{words} words")
            if app_name:
                bits.append(str(app_name))
            if not delivered:
                bits.append("clipboard only")
            meta = NSTextField.labelWithString_("  ·  ".join(bits))
            meta.setFont_(G.rounded_font(11.5))
            meta.setTextColor_(NSColor.secondaryLabelColor())
            meta.setTranslatesAutoresizingMaskIntoConstraints_(False)
            inner.addSubview_(meta)

            # Per-card Copy button. Target = controller; identity via tag.
            tag = self._next_tag
            self._next_tag += 1
            self._rows[tag] = text
            copy = NSButton.buttonWithTitle_target_action_("Copy", self, "copyCard:")
            copy.setTag_(tag)
            copy.setBezelStyle_(1)   # NSBezelStyleRounded
            copy.setFont_(G.rounded_font(12))
            copy.setTranslatesAutoresizingMaskIntoConstraints_(False)
            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                "doc.on.doc", "Copy")
            if img is not None:
                copy.setImage_(img)
                copy.setImagePosition_(NSImageLeft)
            inner.addSubview_(copy)

            # --- Auto Layout: 14pt insets; body left of the Copy button; meta below.
            PADX, PADY = 14.0, 12.0
            copy.topAnchor().constraintEqualToAnchor_constant_(
                inner.topAnchor(), PADY - 2).setActive_(True)
            copy.trailingAnchor().constraintEqualToAnchor_constant_(
                inner.trailingAnchor(), -PADX).setActive_(True)
            copy.widthAnchor().constraintEqualToConstant_(74.0).setActive_(True)

            body.leadingAnchor().constraintEqualToAnchor_constant_(
                inner.leadingAnchor(), PADX).setActive_(True)
            body.topAnchor().constraintEqualToAnchor_constant_(
                inner.topAnchor(), PADY).setActive_(True)
            body.trailingAnchor().constraintEqualToAnchor_constant_(
                copy.leadingAnchor(), -10.0).setActive_(True)

            meta.leadingAnchor().constraintEqualToAnchor_(body.leadingAnchor()).setActive_(True)
            meta.trailingAnchor().constraintEqualToAnchor_(body.trailingAnchor()).setActive_(True)
            meta.topAnchor().constraintEqualToAnchor_constant_(
                body.bottomAnchor(), 6.0).setActive_(True)
            meta.bottomAnchor().constraintEqualToAnchor_constant_(
                inner.bottomAnchor(), -PADY).setActive_(True)
            return container

        # ---- actions (Obj-C selectors — names match the action strings) --
        def searchChanged_(self, sender):
            # Return / search-commit: filter immediately, cancelling any pending
            # debounced rebuild so the two paths can't both fire.
            if self._filter_timer is not None:
                self._filter_timer.invalidate()
                self._filter_timer = None
            self._rebuild(str(sender.stringValue() or ""))

        def controlTextDidChange_(self, note):
            # Debounce live typing: rebuilding up to 100 blur-backed cards on every
            # keystroke stutters the field, so coalesce to one rebuild ~0.12s after
            # the last keypress. searchChanged_ (Return) remains the immediate path.
            if self._filter_timer is not None:
                self._filter_timer.invalidate()
            self._filter_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                0.12, False, lambda _t: self._rebuild(str(self._search.stringValue() or "")))

        def copyCard_(self, sender):
            txt = self._rows.get(int(sender.tag()))
            if not txt:
                return
            _clip_set(txt)
            sender.setTitle_("Copied")
            # Restore the label after ~1.1s on the main runloop (no UI thread).
            NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                1.1, False, lambda _t: self._restore_copy(sender))

        @objc.python_method
        def _restore_copy(self, sender):
            try:
                sender.setTitle_("Copy")
            except Exception:  # noqa: BLE001
                pass

        def openTranscribe_(self, sender):
            try:
                self._app._show_transcribe_window()
            except Exception:  # noqa: BLE001
                pass

        def clearHistory_(self, sender):
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Clear dictation history?")
            alert.setInformativeText_(
                "This permanently removes all saved dictations from this list. "
                "It doesn't affect anything you've already typed.")
            alert.addButtonWithTitle_("Clear")     # first button -> return 1000
            alert.addButtonWithTitle_("Cancel")
            if alert.runModal() == 1000:            # NSAlertFirstButtonReturn
                clear_history()
                self._reload()

    _HISTORY_CTRL_CLASS = _HistoryController
    return _HISTORY_CTRL_CLASS


# ---------------------------------------------------------------------------
# Floating recording HUD — the waveform "pill" that appears near the bottom of
# the screen while you dictate. Native reimplementation of the redesign mockup:
# a dark frosted capsule with a breathing status dot, an animated 22-bar
# waveform, a running mm:ss timer, and a Stop button, plus a hint line beneath.
#
# Safety properties that MUST hold (this shows while you dictate into another
# app): it is a NON-ACTIVATING floating panel shown with orderFrontRegardless,
# so presenting it never steals key focus — the paste still lands in the app you
# were typing into. Everything here runs on the main thread (driven from
# _MenuActions.applyStatus_, which is already marshalled there) and must never
# raise into the dictation path.
# ---------------------------------------------------------------------------
_HUD_CTRL_CLASS = None


def _hud_controller_class():
    """Lazily build & cache the NSObject subclass that owns the recording HUD
    panel. Deferred AppKit import so CLI paths never touch Cocoa."""
    global _HUD_CTRL_CLASS
    if _HUD_CTRL_CLASS is not None:
        return _HUD_CTRL_CLASS

    import math
    import objc
    from Cocoa import (
        NSObject, NSPanel, NSView, NSVisualEffectView, NSTextField, NSButton,
        NSImage, NSColor, NSFont, NSScreen,
        NSMakeRect, NSBackingStoreBuffered,
        NSTextAlignmentCenter, NSTextAlignmentRight,
        NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    )

    G = _glass()

    # Dark HUD vibrancy material (always-dark frosted look, independent of the
    # system light/dark setting). Fall back to the popover material or the raw
    # enum value on older AppKit.
    try:
        from Cocoa import NSVisualEffectMaterialHUDWindow as _MAT_HUD
    except ImportError:  # pragma: no cover
        _MAT_HUD = 13

    # Float above ordinary windows (and over full-screen apps / every Space).
    try:
        from Cocoa import (
            NSStatusWindowLevel as _LEVEL,
            NSWindowCollectionBehaviorCanJoinAllSpaces as _CB_ALL,
            NSWindowCollectionBehaviorStationary as _CB_STATIONARY,
            NSWindowCollectionBehaviorFullScreenAuxiliary as _CB_FSAUX,
        )
    except ImportError:  # pragma: no cover
        _LEVEL, _CB_ALL, _CB_STATIONARY, _CB_FSAUX = 25, 1, 16, 256

    def _white(a):
        return NSColor.whiteColor().colorWithAlphaComponent_(a)

    def _rgb(r, g, b, a=1.0):
        return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a)

    GREEN = _rgb(0.788, 0.925, 0.431)      # #c9ec6e — the früt accent
    BAR = _rgb(0.745, 0.878, 0.396)        # mid of the mockup's bar gradient

    # --- geometry (see the mockup's RECORDING HUD block) --------------------
    PILL_W, PILL_H, RADIUS = 384.0, 54.0, 27.0
    HINT_H, GAP_V = 16.0, 8.0
    PANEL_W, PANEL_H = PILL_W, PILL_H + GAP_V + HINT_H
    PILL_Y = HINT_H + GAP_V                 # pill sits above the hint line
    NBARS = 22
    WAVE_X, WAVE_W, WAVE_H = 137.0, 129.0, 30.0

    class _HudController(NSObject):
        def initWithApp_(self, app):
            self = objc.super(_HudController, self).init()
            if self is None:
                return None
            self._app = app
            self._built = False
            self._panel = None
            self._bars = []
            self._timer = None
            self._state = "hidden"          # hidden | listening | transcribing
            self._t0 = 0.0
            return self

        # -- build (lazy, first show) ---------------------------------------
        def _build(self):
            if self._built:
                return
            style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, PANEL_W, PANEL_H), style,
                NSBackingStoreBuffered, False)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setHasShadow_(True)               # server shadow follows the pill shape
            panel.setLevel_(_LEVEL)
            panel.setFloatingPanel_(True)
            panel.setBecomesKeyOnlyIfNeeded_(True)
            panel.setHidesOnDeactivate_(False)
            panel.setReleasedWhenClosed_(False)
            try:
                panel.setCollectionBehavior_(_CB_ALL | _CB_STATIONARY | _CB_FSAUX)
            except Exception:  # noqa: BLE001
                pass
            self._panel = panel

            content = panel.contentView()

            # The pill: a behind-window frosted capsule (blurs whatever is under it).
            pill = NSVisualEffectView.alloc().initWithFrame_(
                NSMakeRect(0, PILL_Y, PILL_W, PILL_H))
            pill.setBlendingMode_(G.BLEND_BEHIND)
            pill.setState_(G.STATE_ACTIVE)
            try:
                pill.setMaterial_(_MAT_HUD)
            except Exception:  # noqa: BLE001
                pass
            # Force the dark frosted look regardless of the system light/dark
            # setting (the mockup pill is always dark charcoal glass). Vibrant-dark
            # also makes the white-on-dark text and the stop button read correctly.
            try:
                from Cocoa import NSAppearance
                ap = NSAppearance.appearanceNamed_("NSAppearanceNameVibrantDark")
                if ap is not None:
                    pill.setAppearance_(ap)
            except Exception:  # noqa: BLE001
                pass
            pl = G.round_layer(pill, RADIUS, mask=True)
            if pl is not None:
                pl.setBorderWidth_(1.0)
                pl.setBorderColor_(_white(0.16).CGColor())
            content.addSubview_(pill)
            self._pill = pill

            # status dot (breathing green) with a soft glow.
            dot = NSView.alloc().initWithFrame_(
                NSMakeRect(20, (PILL_H - 9) / 2.0, 9, 9))
            dot.setWantsLayer_(True)
            dl = dot.layer()
            if dl is not None:
                dl.setCornerRadius_(4.5)
                dl.setBackgroundColor_(GREEN.CGColor())
                dl.setMasksToBounds_(False)
                dl.setShadowColor_(GREEN.CGColor())
                dl.setShadowRadius_(5.0)
                dl.setShadowOpacity_(0.9)
                dl.setShadowOffset_(_zero_size())
            pill.addSubview_(dot)
            self._dot = dot

            # status label ("Listening" / "Transcribing").
            label = NSTextField.labelWithString_("Listening")
            label.setFrame_(NSMakeRect(38, (PILL_H - 18) / 2.0, 84, 18))
            label.setFont_(G.rounded_font(12.5, 0.3))
            label.setTextColor_(_white(0.75))
            pill.addSubview_(label)
            self._label = label

            # waveform: NBARS thin bars, animated in tick_.
            wave = NSView.alloc().initWithFrame_(
                NSMakeRect(WAVE_X, (PILL_H - WAVE_H) / 2.0, WAVE_W, WAVE_H))
            pill.addSubview_(wave)
            self._bars = []
            for i in range(NBARS):
                bar = NSView.alloc().initWithFrame_(
                    NSMakeRect(i * 6.0, WAVE_H / 2.0 - 4, 3, 8))
                bar.setWantsLayer_(True)
                bl = bar.layer()
                if bl is not None:
                    bl.setCornerRadius_(1.5)
                    bl.setBackgroundColor_(BAR.CGColor())
                wave.addSubview_(bar)
                self._bars.append(bar)
            self._wave = wave

            # running timer (mm:ss, tabular figures so it doesn't jitter).
            tfont = None
            try:
                tfont = NSFont.monospacedDigitSystemFontOfSize_weight_(13, 0.3)
            except Exception:  # noqa: BLE001
                tfont = G.rounded_font(13, 0.3)
            tlab = NSTextField.labelWithString_("0:00")
            tlab.setFrame_(NSMakeRect(281, (PILL_H - 18) / 2.0, 38, 18))
            tlab.setFont_(tfont)
            tlab.setTextColor_(_white(0.82))
            tlab.setAlignment_(NSTextAlignmentRight)
            pill.addSubview_(tlab)
            self._time = tlab

            # Stop button — ends the current capture (safe in hold + toggle).
            stop = self._make_stop_button()
            stop.setFrame_(NSMakeRect(334, (PILL_H - 30) / 2.0, 30, 30))
            pill.addSubview_(stop)
            self._stop = stop

            # hint line beneath the pill.
            hint = NSTextField.labelWithString_(self._hint_text())
            hint.setFrame_(NSMakeRect(0, 0, PILL_W, HINT_H))
            hint.setFont_(G.rounded_font(11.5, 0.0))
            hint.setTextColor_(_white(0.5))
            hint.setAlignment_(NSTextAlignmentCenter)
            self._apply_hint_shadow(hint)
            content.addSubview_(hint)
            self._hint = hint

            self._built = True

        def _make_stop_button(self):
            img = None
            try:
                img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                    "stop.fill", "Stop")
            except Exception:  # noqa: BLE001
                img = None
            if img is not None:
                btn = NSButton.buttonWithImage_target_action_(img, self, "stop:")
            else:
                btn = NSButton.buttonWithTitle_target_action_("■", self, "stop:")
            btn.setBordered_(False)
            try:
                btn.setContentTintColor_(_white(0.85))
            except Exception:  # noqa: BLE001
                pass
            btn.setWantsLayer_(True)
            bl = btn.layer()
            if bl is not None:
                bl.setCornerRadius_(15.0)
                bl.setBackgroundColor_(_white(0.08).CGColor())
                bl.setBorderWidth_(1.0)
                bl.setBorderColor_(_white(0.16).CGColor())
            btn.setToolTip_("Stop")
            return btn

        def _hint_text(self):
            sym = {
                "alt_r": "⌥", "alt_l": "⌥", "alt": "⌥",
                "ctrl_r": "⌃", "ctrl_l": "⌃", "ctrl": "⌃",
                "cmd_r": "⌘", "cmd_l": "⌘", "cmd": "⌘",
                "shift": "⇧", "fn": "fn",
            }.get(str(self._app.hotkey_name).lower(), str(self._app.hotkey_name))
            verb = ("Release %s to insert" if self._app.cfg.get("mode") == "hold"
                    else "Tap %s to stop") % sym
            return "%s · say “never mind” to undo" % verb

        def _apply_hint_shadow(self, field):
            # A soft dark shadow keeps the low-alpha hint legible over both light
            # and dark backgrounds (it floats over whatever app you're in).
            try:
                from Cocoa import (
                    NSShadow, NSAttributedString, NSColor as _C,
                    NSShadowAttributeName, NSFontAttributeName,
                    NSForegroundColorAttributeName, NSParagraphStyleAttributeName,
                    NSMutableParagraphStyle,
                )
                sh = NSShadow.alloc().init()
                sh.setShadowColor_(_C.blackColor().colorWithAlphaComponent_(0.6))
                sh.setShadowBlurRadius_(3.0)
                sh.setShadowOffset_(_zero_size())
                para = NSMutableParagraphStyle.alloc().init()
                para.setAlignment_(NSTextAlignmentCenter)
                attrs = {
                    NSFontAttributeName: field.font(),
                    NSForegroundColorAttributeName: _white(0.55),
                    NSShadowAttributeName: sh,
                    NSParagraphStyleAttributeName: para,
                }
                field.setAttributedStringValue_(
                    NSAttributedString.alloc().initWithString_attributes_(
                        field.stringValue(), attrs))
            except Exception:  # noqa: BLE001
                pass

        # -- placement -------------------------------------------------------
        def _position(self):
            screen = NSScreen.mainScreen()
            if screen is None:
                screens = NSScreen.screens()
                screen = screens[0] if screens else None
            if screen is None:
                return
            vf = screen.visibleFrame()
            x = vf.origin.x + (vf.size.width - PANEL_W) / 2.0
            y = vf.origin.y + 24.0
            self._panel.setFrameOrigin_((x, y))

        # -- state transitions (main thread) --------------------------------
        def showListening(self):
            try:
                self._build()
                fresh = self._state == "hidden"
                self._state = "listening"
                self._label.setStringValue_("Listening")
                self._wave.setAlphaValue_(1.0)
                if fresh:
                    self._t0 = time.monotonic()
                    self._time.setStringValue_("0:00")
                    self._position()
                    self._panel.orderFrontRegardless()
                self._ensure_timer()
            except Exception:  # noqa: BLE001
                pass

        def showTranscribing(self):
            try:
                if not self._built or self._state == "hidden":
                    # Transcribing a queued clip while nothing is actively
                    # recording — surface the pill so state stays visible.
                    self._build()
                    self._t0 = time.monotonic()
                    self._position()
                    self._panel.orderFrontRegardless()
                self._state = "transcribing"
                self._label.setStringValue_("Transcribing")
                self._wave.setAlphaValue_(0.5)
                self._ensure_timer()
            except Exception:  # noqa: BLE001
                pass

        def hideHud(self):
            try:
                self._state = "hidden"
                self._stop_timer()
                if self._panel is not None:
                    self._panel.orderOut_(None)
            except Exception:  # noqa: BLE001
                pass

        # -- animation -------------------------------------------------------
        def _ensure_timer(self):
            if self._timer is not None:
                return
            from Cocoa import NSTimer
            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0 / 30.0, self, "tick:", None, True)

        def _stop_timer(self):
            if self._timer is not None:
                try:
                    self._timer.invalidate()
                except Exception:  # noqa: BLE001
                    pass
                self._timer = None

        def tick_(self, _timer):
            try:
                t = time.monotonic()
                listening = self._state == "listening"
                base, amp, speed = (6.0, 22.0, 7.0) if listening else (5.0, 6.0, 3.0)
                for i, bar in enumerate(self._bars):
                    h = base + amp * (0.5 + 0.5 * math.sin(t * speed + i * 0.55))
                    bar.setFrame_(NSMakeRect(i * 6.0, WAVE_H / 2.0 - h / 2.0, 3, h))
                # breathing status dot
                self._dot.setAlphaValue_(0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 3.0)))
                # running clock
                secs = max(0, int(t - self._t0))
                self._time.setStringValue_("%d:%02d" % (secs // 60, secs % 60))
            except Exception:  # noqa: BLE001
                pass

        # -- actions ---------------------------------------------------------
        def stop_(self, _sender):
            try:
                self._app._end()
            except Exception:  # noqa: BLE001
                pass

    _HUD_CTRL_CLASS = _HudController
    return _HUD_CTRL_CLASS


def _zero_size():
    from Cocoa import NSMakeSize
    return NSMakeSize(0.0, 0.0)


class FlowApp:
    def __init__(self, cfg: dict, transcriber):
        self.cfg = cfg
        self.transcriber = transcriber
        self.recorder = Recorder()
        self.hotkey_name = cfg["hotkey"]
        self.target_vks = resolve_target_vks(cfg["hotkey"])
        self.debug = bool(cfg.get("debug", False))
        self._last_toggle = 0.0
        # When matching a bare modifier via kCGEventFlagsChanged we only get the
        # keycode, not press-vs-release, so we track whether our key is "down".
        self._key_down = False
        self._tap = None          # CFMachPort for the Quartz tap
        self._tap_source = None   # its run-loop source (needed to detach on rebuild)
        self._loop = None         # the main CFRunLoop (set in run())
        self._last_recover = 0.0  # debounce for _recover_after_wake
        self._tap_disabled_streak = 0  # consecutive watchdog checks finding it dead
        self._activity_token = None    # NSActivity assertion (App Nap opt-out)
        self._wake_observer = None     # NSWorkspace DidWake token
        self._wake_observer2 = None    # NSWorkspace ScreensDidWake token
        # One lock guards the recording/processing state transitions, which are
        # touched from both the Quartz callback thread and the worker thread.
        self._state_lock = threading.RLock()
        self._record_started = 0.0   # monotonic time the current capture began
        self._warned_capture = False  # played the pre-cap warning for this capture?
        self._MAX_RECORD_SECONDS = float(cfg.get("max_record_seconds", 120))
        self._warn_before = float(cfg.get("warn_before_max_seconds", 10))
        self._max_processing = float(cfg.get("max_processing_seconds", 120))
        # Finished captures wait here for the SINGLE transcription worker. Decoupling
        # capture from transcription is the key reliability fix: pressing the hotkey
        # while a previous clip is still transcribing now starts a NEW recording and
        # queues it, instead of being silently dropped (the biggest way whole
        # dictations used to be lost). The worker serializes transcription + paste so
        # two clips never overlap.
        self._work_q: queue.Queue = queue.Queue()
        self._processing_started: float | None = None  # set while the worker runs
        self._proc_warned = False
        # Auto-learn-from-edits: a handle on the field we last pasted into, so we
        # can diff your correction against it. None when nothing is pending.
        self._pending_learn: dict | None = None
        # Voice-undo ("never mind"): a stack of the EXACT strings we inserted, most
        # recent last. Saying an undo phrase pops the top and backspaces over it.
        self._undo_stack: list[str] = []
        # "Transcribe an audio file" window (built lazily on first open).
        self._transcribe_ctrl = None
        # "History" window — the app's home page (built lazily on first open).
        self._history_ctrl = None
        # Serialize model access: the mic worker and the file-transcribe window must
        # never call transcribe() on the same model concurrently.
        self._transcribe_lock = threading.Lock()
        # Menu-bar ("real app") mode: on when launched as frutflow.app rather than
        # from an interactive terminal. LaunchServices sets __CFBundleIdentifier
        # for bundle launches; a plain `python flow.py` does not. run() adds a
        # second, independent NSBundle check. `--app` forces it on (testing);
        # `--no-menubar` forces the classic terminal loop even inside the bundle.
        self._menubar_forced_off = "--no-menubar" in sys.argv
        self._app_mode = (not self._menubar_forced_off) and (
            "--app" in sys.argv
            or os.environ.get("__CFBundleIdentifier", "") == APP_BUNDLE_ID
        )
        self._status_item = None   # NSStatusItem (menu-bar glyph)
        self._hud = None           # _HudController (floating waveform pill; lazy)
        self._status_line = None   # disabled NSMenuItem showing the state text
        self._menu = None          # NSMenu (retained so it isn't GC'd)
        self._menu_target = None   # _MenuActions instance (retained; action target)
        self._tap_ok = True        # False if Input Monitoring not yet granted

    def _set_status(self, glyph: str, label: str) -> None:
        """Reflect the dictation state in the menu bar. No-op unless in app mode.
        Marshals the UI update to the main thread (safe from the worker thread)
        and must NEVER raise into the dictation path."""
        if not self._app_mode or self._menu_target is None:
            return
        try:
            self._menu_target.performSelectorOnMainThread_withObject_waitUntilDone_(
                "applyStatus:", [glyph, label], False)
        except Exception:  # noqa: BLE001
            pass

    def _hud_apply(self, glyph: str) -> None:
        """Drive the floating recording HUD from the dictation state glyph. Runs
        on the MAIN THREAD only (called from _MenuActions.applyStatus_). Builds
        the HUD lazily and must never raise into the dictation path.
          🔴 -> listening   ⏳ -> transcribing   anything else -> hidden."""
        if not self.cfg.get("show_hud", True):
            if self._hud is not None:
                try:
                    self._hud.hideHud()
                except Exception:  # noqa: BLE001
                    pass
            return
        try:
            if self._hud is None:
                self._hud = _hud_controller_class().alloc().initWithApp_(self)
            if glyph == "🔴":
                self._hud.showListening()
            elif glyph == "⏳":
                self._hud.showTranscribing()
            else:
                self._hud.hideHud()
        except Exception:  # noqa: BLE001
            pass

    # -- recording lifecycle -------------------------------------------------

    def _begin(self) -> None:
        with self._state_lock:
            if self.recorder.recording:
                return
            try:
                self.recorder.start()
            except Exception as e:  # noqa: BLE001  mic failed even after reinit
                print(f"[flow] could not open the microphone: {e} — try again "
                      "in a moment (if it persists, check System Settings ▸ "
                      "Privacy ▸ Microphone or restart Wispr DIY).", flush=True)
                play("Basso", self.cfg)
                return
            self._record_started = time.monotonic()
            self._warned_capture = False
        play("Tink", self.cfg)
        print("[flow] ● recording...")
        self._set_status("🔴", "● Listening…")
        # Lock in any edit you made to the LAST dictation — but do it OFF the Quartz
        # tap thread and AFTER the mic is already capturing. An Accessibility read of
        # the previous app can block, and doing it before recorder.start() (as we used
        # to) could clip your first words or stall the tap enough that macOS disables
        # it. Reconcile is idempotent + lock-guarded, so a background run is safe.
        if self.cfg.get("learn_from_edits", True):
            threading.Thread(target=self._reconcile_edit_learning,
                             daemon=True).start()

    def _end(self) -> None:
        with self._state_lock:
            if not self.recorder.recording:
                return
            audio = self.recorder.stop()
        play("Pop", self.cfg)
        if audio is None or len(audio) / SAMPLE_RATE < self.cfg["min_seconds"]:
            print("[flow] (too short, ignored)")
            return
        # Hand the clip to the single transcription worker and return immediately —
        # capture is never blocked by a slow transcribe.
        self._work_q.put(audio)
        depth = self._work_q.qsize()
        if depth > 1:
            print(f"[flow] queued — {depth} clips waiting to transcribe.")
            # Distinct cue so you know earlier text is still on its way.
            play("Morse", self.cfg, volume=self.cfg.get("ding_volume", 0.25))

    def _transcription_worker(self) -> None:
        """Single long-lived consumer of the capture queue. Serializes transcription
        and pasting so two clips never overlap, without ever blocking capture."""
        while True:
            audio = self._work_q.get()
            try:
                if audio is None:
                    continue   # shutdown sentinel (unused today)
                with self._state_lock:
                    self._processing_started = time.monotonic()
                    self._proc_warned = False
                self._process(audio)
            except Exception as e:  # noqa: BLE001  never let the worker thread die
                print(f"[flow] worker error: {e}", flush=True)
            finally:
                with self._state_lock:
                    self._processing_started = None
                # Reflect the resulting state in the menu bar (cosmetic only).
                if self.recorder.recording:
                    self._set_status("🔴", "● Listening…")
                elif not self._work_q.empty():
                    self._set_status("⏳", "● Transcribing…")
                else:
                    self._set_status("🎙️", "● Idle")
                self._work_q.task_done()

    def _process(self, audio: np.ndarray) -> None:
        text = ""
        recorded = False   # guard: record each dictation to history at most once
        try:
            print("[flow] transcribing...")
            self._set_status("⏳", "● Transcribing…")
            # Bias the model toward YOUR vocabulary. "hotwords" is the reliable
            # lever (short curated list); "prompt" is the legacy initial_prompt
            # blob. Ignored by the Parakeet backend (no decoder biasing), but the
            # fuzzy corrector in clean() fixes names regardless of engine.
            learn = self.cfg.get("learn_vocab", True)
            mode = self.cfg.get("vocab_biasing", "hotwords")
            prompt = hotwords = None
            if learn and mode == "prompt":
                prompt = build_learned_prompt(self.cfg)
            elif learn and mode == "hotwords":
                hotwords = build_hotwords(self.cfg)
            # Only pay for context capture when the LLM formatter will use it.
            context = ({"app": _focused_app_name()}
                       if self.cfg.get("cleanup") == "llm" else None)
            with self._transcribe_lock:   # never overlap with the file-transcribe window
                raw = self.transcriber.transcribe(audio, prompt=prompt,
                                                  hotwords=hotwords)
            text = clean(raw, self.cfg, context)
            if not text:
                print("[flow] (no speech detected)")
                return
            print(f"[flow] → {text}")
            # Voice undo. A "never mind" phrase retracts the sentence spoken right
            # before it — INLINE, so you can talk, say "actually never mind", and keep
            # going in one breath (only that sentence is dropped from what's typed). If
            # the phrase is the whole utterance or at the very start, it instead
            # retracts the PREVIOUS pasted dictation (backspacing it away).
            if self.cfg.get("undo_enabled", True):
                kept, prev_delete = apply_undo(text, self.cfg)
                if prev_delete or kept != text:
                    self._apply_undo_result(kept, prev_delete)
                    return
            # Wispr-style natural merge: lead with a space so the dictation
            # doesn't glue onto whatever word is already left of the cursor.
            to_insert = (" " + text) if self.cfg.get("auto_space", True) else text
            delivered = insert_text(to_insert, self.cfg)
            # Close the perception loop: a subtle (quiet) cue when text actually
            # lands, a distinct, louder one when it only made it to the clipboard.
            if delivered:
                self._remember_insertion(to_insert)   # so a later "never mind" can delete it
                record_history(text, app=_focused_app_name(), delivered=True)
                recorded = True
                play("Glass", self.cfg, volume=self.cfg.get("ding_volume", 0.25))
                if self.cfg.get("learn_from_edits", True):
                    # Finalize the PREVIOUS paste's pending learn before arming this
                    # one, so a rapid burst of dictations can't drop a correction.
                    self._reconcile_edit_learning()
                    # Arm the auto-learner on the field we just pasted into, so any
                    # fix you make becomes a learned correction (no manual --correct).
                    self._arm_edit_learning(text)
            else:
                # insert_text fell back to clipboard-only: still worth recording.
                record_history(text, app=_focused_app_name(), delivered=False)
                recorded = True
                play("Basso", self.cfg)
            # Learn your vocabulary from the FINAL text (after delivery, so it
            # never delays the paste). No model retraining — just word stats that
            # feed biasing on the next dictation.
            if learn:
                try:
                    learn_vocab(text)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001  never silently eat a recording
            print(f"[flow] ERROR while processing: {e}")
            # Last-ditch: leave the transcript on the clipboard so the user can
            # still paste it manually instead of losing their words.
            if text:
                try:
                    _clip_set(text)
                    print("[flow] (left text on clipboard — press Cmd-V)")
                except Exception:  # noqa: BLE001
                    pass
                # Record even on the error path — unless the delivered/clipboard
                # branch above already recorded this dictation (an exception raised
                # AFTER the record, e.g. in edit-learning, must not double-record).
                # app=None: _focused_app_name may be unhappy here; record never raises.
                if not recorded:
                    record_history(text, app=None, delivered=False)

    def _remember_insertion(self, inserted: str) -> None:
        """Push the EXACT string we just inserted onto the undo stack (bounded), so a
        later 'never mind' can backspace it away."""
        with self._state_lock:
            self._undo_stack.append(inserted)
            if len(self._undo_stack) > 25:
                self._undo_stack.pop(0)

    def _apply_undo_result(self, kept: str, prev_delete: int) -> None:
        """Carry out an utterance that contained a 'never mind': retract `prev_delete`
        previously-pasted dictations (backspace), then type the `kept` remainder (the
        continuation after an inline retraction), if any."""
        print(f"[flow] ↩︎ never mind — retract {prev_delete} prior dictation(s); "
              f"keep {kept!r}", flush=True)
        for _ in range(prev_delete):
            if not self._undo_last_insertion():   # pops stack, AX-verifies, backspaces
                break
        if not kept.strip():
            return
        # Inline retraction with a continuation to type. Give a distinct cue that the
        # retraction registered when we didn't already backspace (which dinged).
        if prev_delete == 0:
            play("Bottle", self.cfg, volume=self.cfg.get("ding_volume", 0.25))
        to_insert = (" " + kept) if self.cfg.get("auto_space", True) else kept
        if insert_text(to_insert, self.cfg):
            self._remember_insertion(to_insert)
            record_history(kept, app=_focused_app_name(), delivered=True)
            play("Glass", self.cfg, volume=self.cfg.get("ding_volume", 0.25))
            if self.cfg.get("learn_from_edits", True):
                self._reconcile_edit_learning()
                self._arm_edit_learning(kept)
            if self.cfg.get("learn_vocab", True):
                try:
                    learn_vocab(kept)
                except Exception:  # noqa: BLE001
                    pass
        else:
            play("Basso", self.cfg)

    def _undo_last_insertion(self) -> bool:
        """Voice command 'never mind': delete the most recent dictation by
        backspacing over the exact text we inserted. Returns True if it deleted.

        Refuses (rather than risk eating your other text) when it can tell the
        inserted span is no longer sitting untouched at the cursor — e.g. you edited
        the pasted text in place (the auto-learn workflow) or moved the cursor. When
        the app doesn't expose its text to Accessibility we can't check, so we delete
        best-effort (counting whole characters, so accents/emoji don't over-delete)."""
        with self._state_lock:
            last = self._undo_stack.pop() if self._undo_stack else None

        def _refuse(msg: str, put_back: bool = True) -> bool:
            print(f"[flow] {msg}")
            if put_back and last:
                with self._state_lock:
                    self._undo_stack.append(last)
            play("Basso", self.cfg, volume=self.cfg.get("ding_volume", 0.25))
            return False

        if not last:
            return _refuse("(never mind — but nothing to undo)", put_back=False)
        # Deleting uses synthetic keystrokes, same as paste: needs Accessibility and
        # is blocked by Secure Input.
        if not _ax_trusted() or _secure_input_active():
            return _refuse("can't undo — Accessibility not granted or Secure Input on.")

        # SAFETY: if the app exposes its text to Accessibility, confirm the words we
        # inserted are STILL the tail of the field (some fields trim the leading/
        # trailing space from their AX value, so accept those variants too). If the
        # tail clearly isn't our dictation anymore (edited in place), refuse rather
        # than backspace into unrelated text. Delete exactly the MATCHED tail so the
        # count is right even when a boundary space was trimmed. When the app is
        # opaque to AX (val is None), delete best-effort.
        tail = last
        val = _ax_read_value(_ax_focused_element())
        if val is not None:
            tail = next((c for c in (last, last.rstrip(), last.lstrip(), last.strip())
                         if c and val.endswith(c)), None)
            if tail is None:
                return _refuse("never mind — the last dictation was changed; leaving it as is.")

        n = _composed_len(tail)   # one Backspace per composed character (macOS rule)
        print(f"[flow] ↩︎ never mind — deleting last dictation ({n} chars).", flush=True)
        _send_backspaces(n)
        # We just removed it, so don't let the auto-learner mine the deleted text.
        with self._state_lock:
            self._pending_learn = None
        # It's gone from the target app, so drop it from History too (best-effort).
        pop_history_matching(tail)
        play("Bottle", self.cfg, volume=self.cfg.get("ding_volume", 0.25))
        return True

    # -- automatic learning from your post-paste edits -----------------------

    def _arm_edit_learning(self, pasted: str) -> None:
        """Remember the field we just pasted into so a later edit can be learned.
        Also schedule a fallback reconcile in case you don't dictate again soon."""
        el = _ax_focused_element()
        if el is None:
            return   # app doesn't expose its text field; nothing to learn from
        with self._state_lock:
            self._pending_learn = {"el": el, "pasted": pasted, "done": False}
        try:
            win = float(self.cfg.get("learn_window_seconds", 20))
            t = threading.Timer(win, self._reconcile_edit_learning)
            t.daemon = True
            t.start()
        except Exception:  # noqa: BLE001
            pass

    def _reconcile_edit_learning(self) -> None:
        """Finalize learning for the last paste: diff your edits, learn the
        phonetic mis-hears. Runs at most once per paste (next-dictation OR timer)."""
        with self._state_lock:
            pend = self._pending_learn
            if not pend or pend.get("done"):
                return
            pend["done"] = True
            self._pending_learn = None
        try:
            learn_from_edit(pend["el"], pend["pasted"])
        except Exception:  # noqa: BLE001  learning must never break dictation
            pass

    # -- logical key events (called from the Quartz tap callback) ------------
    #
    # These stay trivial — they only flip flags / kick off a worker thread, so
    # the Quartz callback never blocks (which would make macOS disable the tap).

    def _on_key_down(self) -> None:
        if self.cfg["mode"] == "hold":
            self._begin()
        else:  # toggle
            now = time.monotonic()
            if now - self._last_toggle < 0.3:   # debounce key auto-repeat
                return
            self._last_toggle = now
            if self.recorder.recording:
                self._end()
            else:
                self._begin()

    def _on_key_up(self) -> None:
        if self.cfg["mode"] == "hold":
            self._end()

    def _watchdog_loop(self) -> None:
        """Background guardian on its own daemon thread. Jobs:

        (1) Warn ~N seconds before the record cap, then auto-stop a runaway capture
            (a missed key-up in hold mode would otherwise record forever).
        (2) SELF-HEAL the keyboard tap. macOS silently disables a CGEventTap
            after sleep/wake, a slow callback, or "secure input" — and once
            disabled it delivers NO events (not even the disable notification),
            so the hotkey goes dead with no sound and no text until restart. We
            poll CGEventTapIsEnabled and re-enable it; if the re-enable doesn't
            STICK (still dead next check), we escalate to a full tap rebuild.
        (3) DETECT SLEEP/WAKE by clock skew: time.monotonic() (mach_absolute_time)
            pauses while the Mac sleeps but time.time() doesn't, so after a
            lid-close/open the wall clock jumps far ahead of the monotonic clock
            between two loop iterations. On detection we rebuild the tap and
            refresh the audio stack — the belt to the wake-notification's braces,
            for the case where the notification never arrives.
        (4) Surface a STUCK transcription: if the worker has been on one clip too
            long it used to fail silently (a "dead hotkey" with a healthy tap). We
            can't force-kill a thread, but we log loudly so it's diagnosable — and
            capture itself keeps working because the queue never blocks recording.
        """
        import Quartz
        warned_dead = False
        prev_wall, prev_mono = time.time(), time.monotonic()
        while True:
            time.sleep(1.0)
            now = time.monotonic()
            # (3) sleep/wake detection via wall-vs-monotonic clock skew
            wall = time.time()
            skew = (wall - prev_wall) - (now - prev_mono)
            prev_wall, prev_mono = wall, now
            if skew > 15.0:
                self._recover_after_wake(
                    f"system slept ~{skew:.0f}s (lid closed?)")
            # (1) pre-cap warning + runaway recording guard
            with self._state_lock:
                held = (self.recorder.recording and self._record_started) or None
                elapsed = (now - self._record_started) if held else 0.0
                warn = (held and not self._warned_capture and self._warn_before > 0
                        and elapsed > self._MAX_RECORD_SECONDS - self._warn_before)
                if warn:
                    self._warned_capture = True
                runaway = bool(held and elapsed > self._MAX_RECORD_SECONDS)
            if warn:
                print(f"[flow] heads up: {self._warn_before:.0f}s until the "
                      f"{self._MAX_RECORD_SECONDS:.0f}s record cap.", flush=True)
                play("Submarine", self.cfg)
            if runaway:
                print("[flow] recording exceeded "
                      f"{self._MAX_RECORD_SECONDS:.0f}s — auto-stopping.",
                      flush=True)
                self._key_down = False
                self._end()
            # (2) tap-health self-heal, escalating to a rebuild if it won't stick
            try:
                if self._tap is None:
                    # A previous rebuild failed and left us with no tap at all —
                    # keep retrying (debounced to every ~5s inside recover).
                    self._recover_after_wake("hotkey tap missing — reinstalling")
                elif not Quartz.CGEventTapIsEnabled(self._tap):
                    self._tap_disabled_streak += 1
                    if self._tap_disabled_streak >= 2:
                        # Re-enabling didn't stick — the tap is a zombie.
                        self._recover_after_wake(
                            "hotkey tap stayed disabled after re-enabling")
                        self._tap_disabled_streak = 0
                    else:
                        Quartz.CGEventTapEnable(self._tap, True)
                        if not warned_dead:
                            print("[flow] hotkey tap was disabled by macOS "
                                  "(sleep/wake?) — re-enabled it.", flush=True)
                            warned_dead = True
                else:
                    warned_dead = False
                    self._tap_disabled_streak = 0
            except Exception:  # noqa: BLE001  never let the watchdog die
                pass
            # (3) stuck-processing guard (loud, once per stuck clip)
            with self._state_lock:
                started = self._processing_started
                stuck = (started is not None
                         and now - started > self._max_processing
                         and not self._proc_warned)
                if stuck:
                    self._proc_warned = True
            if stuck:
                print("[flow] WARNING: a transcription has been running for "
                      f">{self._max_processing:.0f}s — it may be stuck. New "
                      "dictations are still being recorded and queued. If text "
                      "stops appearing, restart Wispr DIY.", flush=True)

    # -- low-level Quartz tap callback ---------------------------------------

    def _handle_event(self, etype, keycode, is_down, flags=None) -> None:
        """Dispatch a decoded keyboard event.

        is_down is True/False for real keyDown/keyUp events. For a bare modifier
        (flagsChanged) the event type can't tell press from release, so is_down
        comes in as None and we resolve it deterministically from `flags`: the
        modifier is "down" iff its mask bit is set in the event's flags. Only if
        flags are somehow unavailable do we fall back to toggling.
        """
        match = keycode in self.target_vks
        if self.debug:
            kindmap = {10: "keyDown", 11: "keyUp", 12: "flagsChanged"}
            kind = kindmap.get(int(etype), str(etype))
            flags_repr = "-" if flags is None else hex(int(flags))
            print(f"[flow][debug] {kind} vk={keycode} is_down={is_down} "
                  f"flags={flags_repr} target_vks={sorted(self.target_vks)} "
                  f"match={match}", flush=True)
        if not match:
            return

        if is_down is None:
            # Bare modifier (flagsChanged): derive down/up from the flag bits.
            mask = _MODIFIER_MASK_BY_VK.get(keycode)
            if mask is not None and flags is not None:
                is_down = bool(int(flags) & mask)
            else:
                # Last-resort fallback: no mask known, so toggle our state.
                self._key_down = not self._key_down
                is_down = self._key_down

        # Self-healing for a missed event: if we think the key is already in the
        # state this event reports, there's nothing to do (re-reading absolute
        # flag bits means a dropped flagsChanged can't leave us stuck inverted).
        if is_down == self._key_down and is_down is not None:
            # Still resync recording lifecycle in hold mode in case state drifted
            # (e.g. key reported up but we're somehow still recording).
            if (self.cfg["mode"] == "hold" and not is_down
                    and self.recorder.recording):
                self._on_key_up()
            return

        self._key_down = is_down
        if is_down:
            self._on_key_down()
        else:
            self._on_key_up()

    # -- tap lifecycle (create / destroy / rebuild) ---------------------------
    #
    # The tap is installed once at startup and REBUILT from scratch whenever the
    # machine wakes from sleep. Merely re-enabling a post-wake tap is not enough:
    # macOS can leave it a "zombie" that reports enabled yet delivers no events —
    # the historical dead-hotkey-after-opening-the-lid bug.

    def _tap_callback(self, proxy, etype, event, refcon):  # noqa: ARG002
        import Quartz
        try:
            et = int(etype)
            # macOS disabled our tap (callback too slow, sleep/wake, secure
            # input). Re-arm it immediately; if that doesn't stick, the watchdog
            # escalates to a full rebuild.
            if et in (int(Quartz.kCGEventTapDisabledByTimeout),
                      int(Quartz.kCGEventTapDisabledByUserInput)):
                print("[flow] tap disabled by macOS — re-enabling.", flush=True)
                if self._tap is not None:
                    Quartz.CGEventTapEnable(self._tap, True)
                return event

            keycode = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode)
            if et == int(Quartz.kCGEventKeyDown):
                self._handle_event(et, keycode, True)
            elif et == int(Quartz.kCGEventKeyUp):
                self._handle_event(et, keycode, False)
            elif et == int(Quartz.kCGEventFlagsChanged):
                # Bare modifier — no down/up. Read the event's flag bits so
                # _handle_event can derive press/release DETERMINISTICALLY from
                # the matching modifier's mask bit, instead of toggling a Python
                # flag that gets stuck inverted if an event is ever missed.
                flags = Quartz.CGEventGetFlags(event)
                self._handle_event(et, keycode, None, flags)
        except Exception as e:  # noqa: BLE001  never let the tap die
            print(f"[flow] tap callback error: {e}", flush=True)
        return event   # listen-only, but return the event unchanged

    def _install_tap(self) -> bool:
        """Create the listen-only Quartz tap and attach it to the stored run loop.
        Returns False when creation fails (== Input Monitoring not granted)."""
        import Quartz
        # Event types we care about: real key down/up (character keys) AND
        # flagsChanged (bare modifiers like Option). CGEventMaskBit(t) == 1 << t;
        # built manually so we don't depend on the helper being exported.
        mask = ((1 << int(Quartz.kCGEventKeyDown))
                | (1 << int(Quartz.kCGEventKeyUp))
                | (1 << int(Quartz.kCGEventFlagsChanged)))
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,           # session-level tap
            Quartz.kCGHeadInsertEventTap,        # see events first
            Quartz.kCGEventTapOptionListenOnly,  # observe only — needs only
                                                 # Input Monitoring, not Accessibility
            mask,
            self._tap_callback,
            None,
        )
        if not tap:
            return False
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        # Nudge the (possibly sleeping) main run loop so it starts servicing the
        # new source right away — matters when the rebuild happens off-thread.
        Quartz.CFRunLoopWakeUp(self._loop)
        self._tap, self._tap_source = tap, source
        return True

    def _teardown_tap(self) -> None:
        """Fully dismantle the current tap (disable, detach from the run loop,
        invalidate the mach port) so a fresh one can take its place."""
        import Quartz
        tap, source = self._tap, self._tap_source
        self._tap = self._tap_source = None
        try:
            if tap is not None:
                Quartz.CGEventTapEnable(tap, False)
            if source is not None and self._loop is not None:
                Quartz.CFRunLoopRemoveSource(self._loop, source,
                                             Quartz.kCFRunLoopCommonModes)
            if tap is not None:
                Quartz.CFMachPortInvalidate(tap)
        except Exception:  # noqa: BLE001  teardown must never take the app down
            pass

    def _tap_enabled(self) -> bool:
        import Quartz
        try:
            return bool(self._tap is not None
                        and Quartz.CGEventTapIsEnabled(self._tap))
        except Exception:  # noqa: BLE001
            return False

    def _recover_after_wake(self, reason: str) -> None:
        """Post-sleep recovery: rebuild the hotkey tap and refresh the audio stack.
        Idempotent and debounced — the wake notification, the watchdog's clock-skew
        detector, and the disabled-streak escalation may all fire for one wake."""
        now = time.monotonic()
        with self._state_lock:
            if now - self._last_recover < 5.0:
                return
            self._last_recover = now
        print(f"[flow] {reason} — rebuilding the hotkey tap.", flush=True)
        try:
            self._teardown_tap()
            ok = self._install_tap() and self._tap_enabled()
            print(f"[flow] hotkey tap rebuilt (enabled={ok}).", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[flow] tap rebuild failed: {e} — the watchdog will retry.",
                  flush=True)
        # Refresh PortAudio so the next recording doesn't open a stale post-sleep
        # audio device (also picks up default-input changes, e.g. AirPods).
        # Only when idle — never yank the stream out from under a live capture.
        with self._state_lock:
            recording = self.recorder.recording
        if not recording:
            try:
                import sounddevice as sd
                sd._terminate()
                sd._initialize()
            except Exception:  # noqa: BLE001  next InputStream open will retry anyway
                pass

    # -- run (direct Quartz CGEventTap, listen-only) -------------------------

    def run(self) -> None:
        import Quartz

        # Populate the modifier flag masks now that Quartz is imported. For a
        # bare modifier we read down/up from the event's flag bits (see
        # _handle_event) instead of toggling a Python flag that gets stuck
        # inverted if an event is ever missed. We map BOTH Option keycodes to
        # the generic Alternate mask: this pyobjc build does not export the
        # device-specific NX_DEVICE*ALT* masks, and since we accept either
        # Option interchangeably, the generic "any Alt is down" bit is exactly
        # the right signal (set while either Option is held, clear when both
        # are released).
        _alt = int(Quartz.kCGEventFlagMaskAlternate)
        _ctrl = int(Quartz.kCGEventFlagMaskControl)
        _cmd = int(Quartz.kCGEventFlagMaskCommand)
        _shift = int(Quartz.kCGEventFlagMaskShift)
        _MODIFIER_MASK_BY_VK.update({
            58: _alt, 61: _alt,        # left / right Option
            59: _ctrl, 62: _ctrl,      # left / right Control
            55: _cmd, 54: _cmd,        # left / right Command
            56: _shift, 60: _shift,    # left / right Shift
        })

        verb = "Hold" if self.cfg["mode"] == "hold" else "Tap"
        backend = self.cfg["transcribe_backend"]
        shown_model = (self.cfg.get("parakeet_model", "") if backend == "parakeet"
                       else self.cfg["model"])
        print("=" * 60)
        print("  Wispr DIY is running.")
        print(f"  {verb} [{self.hotkey_name}] to dictate. Ctrl-C to quit.")
        print(f"  backend={backend} model={shown_model} "
              f"cleanup={self.cfg['cleanup']}")
        feats = []
        if self.cfg.get("learn_vocab", True):
            feats.append(f"biasing={self.cfg.get('vocab_biasing', 'hotwords')}")
        if self.cfg.get("fuzzy_correct", True):
            feats.append("fuzzy-correct=on"
                         + ("" if FUZZY_AVAILABLE else " (libs missing!)"))
        if self.cfg.get("learn_from_edits", True):
            feats.append("auto-learn=on")
        if feats:
            print("  accuracy: " + "  ".join(feats))
        if self.debug:
            print("  [debug] key logging ON — every keypress is logged below.")
        print("=" * 60)
        _trust_probe()
        if self.cfg["insert_method"] == "paste":
            if not _ax_trusted():
                who = "frutflow" if self._app_mode else "the app running this (Terminal)"
                print(f"[flow] WARNING: Accessibility is NOT granted to {who} — "
                      "auto-paste will fail and dictation will be left on the "
                      "clipboard. Grant Accessibility in System Settings ▸ "
                      "Privacy & Security ▸ Accessibility, then Restart the app "
                      "(the grant is cached per-process).",
                      flush=True)
            if _secure_input_active():
                print("[flow] WARNING: Secure Keyboard Entry is ON in some app — "
                      "it blocks synthetic paste. Turn it off (e.g. Terminal ▸ "
                      "Secure Keyboard Entry) to let auto-paste work.", flush=True)

        self._loop = Quartz.CFRunLoopGetCurrent()
        self._tap_ok = self._install_tap()
        if not self._tap_ok:
            who = "frutflow" if self._app_mode else "the Python binary that runs flow.py"
            print(f"[flow] CGEventTapCreate returned NULL — {who} lacks Input "
                  "Monitoring permission. Grant it in System Settings ▸ Privacy & "
                  "Security ▸ Input Monitoring, then Restart the app.", flush=True)
            if not self._app_mode:
                # Classic/terminal mode: no menu bar to fall back to — bail as before.
                return
            # App mode: fall through so the menu bar STILL appears (in a degraded
            # "needs permission" state). The user can grant the permission and pick
            # menu ▸ Restart — a fresh process then re-creates the tap successfully.
        else:
            print(f"[flow] event tap enabled={self._tap_enabled()}", flush=True)

        # Opt out of App Nap so macOS never throttles this (hidden-Terminal-hosted)
        # process's timers/threads. The *AllowingIdleSystemSleep* variant is important:
        # it must NOT keep the Mac awake — only keep us responsive while it is awake.
        try:
            from Foundation import NSProcessInfo
            import Foundation
            self._activity_token = (
                NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                    Foundation.NSActivityUserInitiatedAllowingIdleSystemSleep,
                    "Wispr DIY listens for the dictation hotkey"))
        except Exception:  # noqa: BLE001  cosmetic-level optimization only
            self._activity_token = None

        # SLEEP/WAKE FIX: after lid-close/open, a CGEventTap can come back as a
        # "zombie" — CGEventTapIsEnabled() says True but it never delivers another
        # event, so re-enabling (the watchdog's old trick) can't heal it and the
        # hotkey stays dead until restart. The reliable cure (what Karabiner-class
        # tools do) is to REBUILD the tap on wake. We hook the wake notification;
        # the watchdog's clock-skew detector covers the case where this
        # notification doesn't arrive.
        try:
            from AppKit import NSWorkspace
            nc = NSWorkspace.sharedWorkspace().notificationCenter()

            def _on_wake(_note):
                self._recover_after_wake("mac woke from sleep")

            def _on_screens_wake(_note):
                self._recover_after_wake("displays woke (lid open?)")

            # queue=None => delivered on the posting thread's run loop; CFRunLoop
            # surgery is safe from any thread (verified) and _recover_after_wake is
            # debounced, so both notifications firing for one wake = ONE rebuild.
            # DidWake covers full system wake; ScreensDidWake covers lid-open /
            # display-only wake, which some wake paths post INSTEAD of DidWake — the
            # gap that leaves the tap a zombie "sometimes" after closing the lid.
            self._wake_observer = nc.addObserverForName_object_queue_usingBlock_(
                "NSWorkspaceDidWakeNotification", None, None, _on_wake)
            self._wake_observer2 = nc.addObserverForName_object_queue_usingBlock_(
                "NSWorkspaceScreensDidWakeNotification", None, None, _on_screens_wake)
        except Exception as e:  # noqa: BLE001  watchdog skew-detector still covers us
            print(f"[flow] (wake-notification hook unavailable: {e})", flush=True)

        # Single transcription worker: drains the capture queue so recording is
        # never blocked while a clip transcribes (no more dropped bursty dictation).
        threading.Thread(target=self._transcription_worker, daemon=True).start()

        # Watchdog: pre-cap warning + auto-stop long captures, tap self-heal,
        # stuck-transcription guard, sleep/wake recovery.
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

        # Second, independent app-mode signal: is the running bundle frutflow.app?
        # (Covers the case where __CFBundleIdentifier isn't in the environment.)
        if not self._app_mode and not self._menubar_forced_off:
            try:
                from AppKit import NSBundle
                if NSBundle.mainBundle().bundleIdentifier() == APP_BUNDLE_ID:
                    self._app_mode = True
            except Exception:  # noqa: BLE001
                pass

        if self._app_mode:
            # Real macOS app: drive the main run loop from NSApplication so we can
            # own a menu-bar item. NSApp's loop IS the main CFRunLoop, so the tap
            # (added in kCFRunLoopCommonModes) and the wake observers keep working
            # exactly as before — the menu bar is purely additive.
            print("[flow] launching as a menu-bar app (frutflow.app).", flush=True)
            try:
                self._run_menubar()
            finally:
                self._teardown_tap()
        else:
            # Classic/terminal mode (e.g. `python flow.py` for debugging): plain
            # CFRunLoop so Ctrl-C still quits and stdout logs stay visible.
            try:
                Quartz.CFRunLoopRun()
            except KeyboardInterrupt:
                raise
            finally:
                self._teardown_tap()

    def _transcribe_path(self, path: str):
        """Transcribe an audio FILE with the app's already-loaded engine + the full
        cleanup/vocab pipeline. Returns (text, error_message). Reuses self.transcriber
        (no second model load) and serializes with the live mic path via a lock."""
        audio = _load_audio_file(path)
        if audio is None:
            return None, "Couldn't read that audio file (unsupported format or corrupt)."
        if len(audio) / SAMPLE_RATE < 0.05:
            return None, "That file has essentially no audio."
        prompt = hotwords = None
        if self.cfg.get("learn_vocab", True):
            if self.cfg.get("vocab_biasing", "hotwords") == "prompt":
                prompt = build_learned_prompt(self.cfg)
            elif self.cfg.get("vocab_biasing", "hotwords") == "hotwords":
                hotwords = build_hotwords(self.cfg)
        with self._transcribe_lock:
            raw = self.transcriber.transcribe(audio, prompt=prompt, hotwords=hotwords)
        return (clean(raw, self.cfg) or ""), None

    def _show_history_window(self) -> None:
        """Open (or re-focus) the History window — the app's home page. Built
        lazily; re-reads history.json and refreshes on every show()."""
        try:
            if self._history_ctrl is None:
                self._history_ctrl = (
                    _history_controller_class().alloc().initWithApp_(self))
            self._history_ctrl.show()
            print("[flow] history window opened.", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[flow] couldn't open the history window: {e}", flush=True)

    def _show_transcribe_window(self) -> None:
        """Open (or re-focus) the 'Transcribe an audio file' window. Built lazily."""
        try:
            if self._transcribe_ctrl is None:
                self._transcribe_ctrl = (
                    _transcribe_controller_class().alloc().initWithApp_(self))
            self._transcribe_ctrl.show()
            print("[flow] transcribe window opened.", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[flow] couldn't open the transcribe window: {e}", flush=True)

    def _run_menubar(self) -> None:
        """Set up the menu-bar status item and run the NSApplication event loop.
        Blocks until the app is told to quit. Only called in app mode. Every
        interaction the old Terminal window + .command files gave you lives here:
        state at a glance (glyph), Teach a Word, Restart, Open Log, Quit."""
        from Cocoa import (NSApplication, NSApplicationActivationPolicyAccessory,
                           NSStatusBar, NSVariableStatusItemLength,
                           NSMenu, NSMenuItem)
        app = NSApplication.sharedApplication()
        # Accessory = menu-bar item, no Dock icon, no window. Unlike a true
        # LSBackgroundOnly agent, an accessory app can still present the mic
        # prompt and appears in the Privacy & Security lists.
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        # NOW that we're a real app, request microphone access. Doing this before
        # NSApplication existed (in main) never surfaced the TCC dialog and left
        # frutflow off the Microphone list. Off the main thread so the menu bar
        # still paints immediately; the system presents the prompt independently.
        threading.Thread(target=ensure_microphone_access, daemon=True).start()

        target = _menu_actions_class().alloc().initWithApp_(self)
        self._menu_target = target
        # Make `target` the app delegate too, so double-clicking the app (a "reopen"
        # while it's already running) opens the transcribe window.
        app.setDelegate_(target)

        # Initial state: healthy (idle) unless the hotkey tap couldn't be created
        # because Input Monitoring isn't granted to frutflow yet.
        if self._tap_ok:
            glyph, label = "🎙️", "● Idle"
        else:
            glyph, label = "⚠️", "● Needs Input Monitoring — grant, then Restart"

        item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength)
        self._status_item = item
        try:
            item.button().setTitle_(glyph)
        except Exception:  # noqa: BLE001  very old AppKit fallback
            item.setTitle_(glyph)

        menu = NSMenu.alloc().init()
        self._menu = menu

        def _add(title, sel, enabled=True):
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, sel, "")
            if sel is not None:
                it.setTarget_(target)
            it.setEnabled_(enabled)
            menu.addItem_(it)
            return it

        _add("Wispr DIY", None, enabled=False)
        self._status_line = _add(label, None, enabled=False)
        menu.addItem_(NSMenuItem.separatorItem())
        _add("History…", "historyWindow:")
        _add("Transcribe Audio File…", "transcribeFile:")
        _add("Teach a Word…", "teachWord:")
        menu.addItem_(NSMenuItem.separatorItem())
        _add("Restart", "restart:")
        _add("Open Log", "openLog:")
        _add("Privacy Settings…", "openPrivacy:")
        menu.addItem_(NSMenuItem.separatorItem())
        _add("Quit Wispr DIY", "quit:")
        item.setMenu_(menu)

        # We're on the main thread here, so paint the initial state directly.
        app.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PERMISSIONS_HELP = """
macOS permissions (System Settings ▸ Privacy & Security):
  • Microphone        — allow the app running this (Terminal / iTerm / VS Code)
  • Accessibility     — required to paste/type into other apps
  • Input Monitoring  — required for the global hotkey to be detected

Grant these to whatever program launches flow.py (e.g. "Terminal"). After
granting, fully quit and reopen that program so the change takes effect.
"""


def ensure_microphone_access() -> None:
    """Make sure macOS has granted microphone access to THIS process.

    A background launchd agent that opens the mic only through PortAudio does
    not reliably trigger the system permission prompt — it just silently
    receives zeroed-out audio. We explicitly ask via AVFoundation, which shows
    the standard "allow microphone" dialog whenever the permission is still
    undetermined. Once granted, this returns immediately on every later launch.
    """
    try:
        import AVFoundation
    except ModuleNotFoundError:
        return  # AVFoundation not installed; PortAudio will try on its own
    AV = AVFoundation.AVCaptureDevice
    media = AVFoundation.AVMediaTypeAudio
    # 0 notDetermined · 1 restricted · 2 denied · 3 authorized
    status = AV.authorizationStatusForMediaType_(media)
    if status == 3:
        return
    if status in (1, 2):
        print("[flow] microphone access is OFF. Turn it on in System Settings "
              "▸ Privacy & Security ▸ Microphone, then restart Wispr DIY.",
              flush=True)
        return

    # notDetermined -> request access; this pops the macOS permission prompt.
    print("[flow] requesting microphone access — please click 'Allow' on "
          "the macOS prompt...", flush=True)
    done = threading.Event()

    def _handler(granted):  # called on an internal dispatch queue
        done.set()

    try:
        AV.requestAccessForMediaType_completionHandler_(media, _handler)
    except Exception as e:  # noqa: BLE001
        print(f"[flow] could not request mic access: {e}", flush=True)
        return

    # Poll the status too, so we don't depend solely on the async handler.
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if done.is_set() or AV.authorizationStatusForMediaType_(media) != 0:
            break
        time.sleep(0.5)

    if AV.authorizationStatusForMediaType_(media) == 3:
        print("[flow] microphone access granted. ✓", flush=True)
    else:
        print("[flow] microphone not granted yet — you can enable it later "
              "in System Settings ▸ Privacy & Security ▸ Microphone.",
              flush=True)


def _load_audio_file(path: str) -> np.ndarray | None:
    """Load any audio file macOS can read into a 16 kHz mono float32 array — the same
    format the mic path produces. Uses `afconvert` (ships with macOS) to normalize
    sample rate / channels / codec, so wav, aiff, caf, m4a, mp3, etc. all work without
    ffmpeg. Returns None (with a message) if the file is missing or unreadable."""
    import wave
    import tempfile
    src = Path(path).expanduser()
    if not src.exists():
        print(f"[flow] file not found: {src}")
        return None
    tmp = Path(tempfile.gettempdir()) / f"wisprdiy_in_{os.getpid()}.wav"
    try:
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
             str(src), str(tmp)],
            check=True, capture_output=True)
        with wave.open(str(tmp), "rb") as w:
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    except FileNotFoundError:
        print("[flow] 'afconvert' not found (expected on macOS); cannot read this file.")
        return None
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode(errors="ignore").strip()[:200]
        print(f"[flow] could not decode {src.name}: {msg or 'unsupported format'}")
        return None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def transcribe_file(cfg: dict, path: str, *, copy: bool = False) -> int:
    """Transcribe an existing audio file through the configured engine + the full
    correction pipeline, and print the text. No mic, no paste — handy for dictating
    from a voice memo, cleaning up a recording, or scripting."""
    audio = _load_audio_file(path)
    if audio is None:
        return 1
    dur = len(audio) / SAMPLE_RATE
    if dur < 0.05:
        print("[flow] that file has essentially no audio.")
        return 1
    print(f"[flow] {Path(path).name}: {dur:.1f}s — transcribing with "
          f"'{cfg.get('transcribe_backend', 'parakeet')}'...", file=sys.stderr,
          flush=True)
    try:
        tb = build_transcriber(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[flow] could not start the transcriber: {e}")
        return 1

    # Same vocabulary biasing the live path uses (ignored by Parakeet, used by whisper).
    prompt = hotwords = None
    if cfg.get("learn_vocab", True):
        if cfg.get("vocab_biasing", "hotwords") == "prompt":
            prompt = build_learned_prompt(cfg)
        elif cfg.get("vocab_biasing", "hotwords") == "hotwords":
            hotwords = build_hotwords(cfg)

    raw = tb.transcribe(audio, prompt=prompt, hotwords=hotwords)
    text = clean(raw, cfg)
    if not text:
        print("[flow] (no speech detected)", file=sys.stderr)
        return 0
    print(text)                         # transcript on stdout, clean for piping
    if copy:
        _clip_set(text)
        print("[flow] ✓ copied to clipboard.", file=sys.stderr)
    return 0


def compare_engines(cfg: dict, seconds: float = 6.0) -> int:
    """Record one clip and transcribe it with BOTH Parakeet and faster-whisper so you
    can A/B them on your own voice (especially quiet/whispered speech) before deciding
    which to keep as `transcribe_backend`. Latency is the trustworthy signal; accuracy
    differences on your real speech are what to judge."""
    ensure_microphone_access()
    rec = Recorder()
    print(f"[flow] recording {seconds:.0f}s — start speaking (or WHISPER) now...",
          flush=True)
    rec.start()
    time.sleep(seconds)
    audio = rec.stop()
    if audio is None or audio.size == 0:
        print("[flow] no audio captured (mic permission?).")
        return 1
    print(f"[flow] captured {len(audio)/SAMPLE_RATE:.1f}s. Loading engines...\n",
          flush=True)

    norm = dict(
        normalize=cfg.get("normalize_audio", True),
        normalize_peak=cfg.get("normalize_peak", 0.95),
        normalize_method=cfg.get("normalize_method", "rms"),
        normalize_rms_dbfs=cfg.get("normalize_rms_dbfs", -20.0),
    )
    engines = []
    try:
        engines.append((
            f"parakeet ({cfg.get('parakeet_model', '').split('/')[-1]})",
            ParakeetTranscriber(
                cfg.get("parakeet_model", "mlx-community/parakeet-tdt-0.6b-v2"),
                cfg["language"], warmup=True, **norm)))
    except Exception as e:  # noqa: BLE001
        print(f"[flow] parakeet unavailable: {e}")
    try:
        engines.append((
            f"faster-whisper ({cfg['model']})",
            LocalTranscriber(
                cfg["model"], cfg["compute_type"], cfg["language"],
                vad_filter=cfg.get("vad_filter", False),
                beam_size=cfg.get("beam_size", 5),
                initial_prompt=cfg.get("initial_prompt", ""),
                cpu_threads=cfg.get("cpu_threads", 0), **norm)))
    except Exception as e:  # noqa: BLE001
        print(f"[flow] faster-whisper unavailable: {e}")

    clean_cfg = {**cfg, "cleanup": "basic"}   # never hit the network for a dry run
    print()
    for name, eng in engines:
        try:
            eng.transcribe(audio)                    # warm-up run
            t0 = time.perf_counter()
            raw = eng.transcribe(audio)              # timed (warm) run
            dt = time.perf_counter() - t0
            cleaned = clean(raw, clean_cfg)
            print(f"── {name}  [{dt:.2f}s]")
            print(f"     raw    : {raw or '(nothing)'}")
            print(f"     cleaned: {cleaned or '(nothing)'}\n")
        except Exception as e:  # noqa: BLE001
            print(f"── {name}: ERROR {e}\n")
    print("[flow] tip: run this a few times with different quiet/whispered lines. "
          "To switch engines, set \"transcribe_backend\" in "
          f"{CONFIG_PATH} to \"parakeet\" or \"local\".")
    return 0


def main() -> int:
    # When launched as the app, stdout/stderr are redirected to flow.log, where
    # Python block-buffers them — so status lines only appear minutes later. Make
    # them line-buffered so the log reflects what's happening in real time.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:  # noqa: BLE001
            pass
    parser = argparse.ArgumentParser(description="Wispr DIY — local voice dictation")
    parser.add_argument("--setup", action="store_true",
                        help="write default config + print permission help")
    parser.add_argument("--list-devices", action="store_true",
                        help="list audio input devices and exit")
    parser.add_argument("--correct", nargs=2, metavar=("HEARD", "CORRECT"),
                        help='teach a correction, e.g. --correct "fruit" "früt"')
    parser.add_argument("--show-learning", action="store_true",
                        help="print your learned vocabulary + corrections and exit")
    parser.add_argument("--try", dest="try_text", metavar="TEXT",
                        help="run TEXT through the correction pipeline and print "
                             "the result (handy for sanity-checking fuzzy fixes)")
    parser.add_argument("--compare", nargs="?", type=float, const=6.0,
                        metavar="SECONDS",
                        help="record SECONDS (default 6) from the mic, then transcribe "
                             "the SAME clip with BOTH Parakeet and faster-whisper and "
                             "print them side by side. Use it to A/B the engines on "
                             "your own quiet/whispered speech before choosing a default.")
    parser.add_argument("--transcribe", metavar="FILE",
                        help="transcribe an existing audio FILE (wav/aiff/m4a/mp3/…) "
                             "with your configured engine and print the text. Add "
                             "--copy to also put it on the clipboard.")
    parser.add_argument("--copy", action="store_true",
                        help="with --transcribe: also copy the transcript to the clipboard")
    parser.add_argument("--app", action="store_true",
                        help="force menu-bar app mode (normally auto-detected when "
                             "launched as frutflow.app)")
    parser.add_argument("--no-menubar", action="store_true",
                        help="force the classic terminal run loop even inside the "
                             ".app bundle (no menu-bar item)")
    args = parser.parse_args()

    if args.setup:
        write_default_config()
        print(PERMISSIONS_HELP)
        return 0

    if args.compare is not None:
        return compare_engines(load_config(), float(args.compare))

    if args.transcribe is not None:
        return transcribe_file(load_config(), args.transcribe, copy=args.copy)

    if args.correct:
        add_correction(args.correct[0], args.correct[1])
        return 0

    if args.show_learning:
        corr = load_corrections()
        vocab = _read_json(VOCAB_PATH, {})
        print(f"[flow] corrections ({len(corr)}) — auto-learned + taught:")
        for h, c in corr.items():
            print(f"    {h!r} -> {c!r}")
        terms = distinctive_terms()
        print(f"[flow] distinctive vocab used for biasing + fuzzy-correct "
              f"({len(terms)}):")
        print("    " + ", ".join(terms) if terms else "    (none yet)")
        print(f"[flow] hotwords sent to the model:")
        print("    " + (build_hotwords(load_config()) or "(none)"))
        top = sorted((vocab or {}).values(),
                     key=lambda e: e.get("count", 0), reverse=True)[:30]
        print(f"[flow] top learned words ({len(vocab or {})} total):")
        print("    " + ", ".join(f"{e.get('form')}({e.get('count')})" for e in top))
        if not FUZZY_AVAILABLE:
            print("[flow] NOTE: rapidfuzz/jellyfish not installed — fuzzy "
                  "correction is disabled. `pip install -r requirements.txt`.")
        return 0

    if args.try_text is not None:
        cfg = load_config()
        # Don't hit the network for a dry run — force the deterministic path.
        if cfg.get("cleanup") == "llm":
            cfg["cleanup"] = "basic"
        print("in : " + args.try_text)
        print("out: " + clean(args.try_text, cfg))
        return 0

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    cfg = load_config()
    try:
        transcriber = build_transcriber(cfg)
    except ModuleNotFoundError as e:
        print(f"[flow] missing dependency: {e.name}\n"
              f"       install requirements first:  pip install -r requirements.txt")
        return 1

    app = FlowApp(cfg, transcriber)
    # The mic TCC prompt can only be presented by a bona-fide app. In classic /
    # terminal mode the host terminal is that app, so ask now. In menu-bar app
    # mode the process isn't an app until NSApplication + activation policy are
    # set, so _run_menubar requests it there instead (asking here would silently
    # no-op and never surface a dialog — the bug that left frutflow unlisted).
    if not app._app_mode:
        ensure_microphone_access()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[flow] bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
