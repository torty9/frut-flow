"""Regression tests for frut Flow's long-running process lifecycle.

These tests intentionally bypass ``FlowApp.__init__`` and ``Recorder.__init__``.
They exercise the coordination logic with small fakes, so the suite never opens a
microphone, creates a Quartz event tap, asks for macOS permissions, or loads an ML
model.
"""

from __future__ import annotations

import gc
import io
import os
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
import weakref
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flow  # noqa: E402  imported after adding the repository root


class WakeRecoveryTests(unittest.TestCase):
    @staticmethod
    def _app():
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._wake_recovery_pending = False
        app._wake_pending_reason = ""
        app._last_recover_wall = 0.0
        app._last_runtime_recover_wall = 0.0
        app.recoveries = []
        app.runtime_recoveries = []
        app._perform_wake_recovery = (
            lambda reason, *, runtime: app.recoveries.append((reason, runtime)))
        app._perform_runtime_recovery = (
            lambda reason: app.runtime_recoveries.append(reason))
        return app

    def test_dark_wake_defers_then_visible_wake_recovers_once(self):
        app = self._app()

        with mock.patch.object(flow, "_display_is_awake", return_value=False), \
                mock.patch.object(flow.time, "time", return_value=100.0), \
                redirect_stdout(io.StringIO()):
            self.assertFalse(app._recover_after_wake("maintenance dark wake"))

        self.assertTrue(app._wake_recovery_pending)
        self.assertEqual(app._wake_pending_reason, "maintenance dark wake")
        self.assertEqual(app.recoveries, [])

        # A visible wake must consume deferred work even if it lands inside the
        # ordinary five-second debounce window. A second clustered notification
        # is then ignored.
        with mock.patch.object(flow.time, "time", side_effect=[101.0, 102.0]):
            self.assertTrue(app._recover_after_wake(
                "display woke", visible=True, runtime=True))
            self.assertFalse(app._recover_after_wake(
                "duplicate display wake", visible=True, runtime=True))

        self.assertFalse(app._wake_recovery_pending)
        self.assertEqual(app._wake_pending_reason, "")
        self.assertEqual(app.recoveries, [("display woke", True)])

    def test_recovery_runs_again_after_debounce_window(self):
        app = self._app()

        with mock.patch.object(flow.time, "time", side_effect=[10.0, 14.9, 15.1]):
            self.assertTrue(app._recover_after_wake(
                "first", visible=True, runtime=False))
            self.assertFalse(app._recover_after_wake(
                "clustered", visible=True, runtime=False))
            self.assertTrue(app._recover_after_wake(
                "later", visible=True, runtime=False))

        self.assertEqual(app.recoveries, [
            ("first", False),
            ("later", False),
        ])

    def test_real_wake_upgrades_recent_tap_only_recovery(self):
        app = self._app()

        with mock.patch.object(flow.time, "time", side_effect=[10.0, 11.0]):
            self.assertTrue(app._recover_after_wake(
                "tap health", visible=True, runtime=False))
            self.assertTrue(app._recover_after_wake(
                "display woke", visible=True, runtime=True))

        self.assertEqual(app.recoveries, [("tap health", False)])
        self.assertEqual(app.runtime_recoveries, ["display woke"])


class CaptureLifecycleTests(unittest.TestCase):
    def test_quick_press_release_preserves_both_fifo_transitions(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._capture_q = queue.SimpleQueue()
        app._capture_requested = False
        app._capture_generation = 0
        app._locked = False
        operations = []
        app._begin = lambda **_kwargs: operations.append("begin")
        app._end = lambda **_kwargs: operations.append("end")

        # Queue both edges before the worker has a chance to inspect the latest
        # desired state. This is the exact fast press/release sequence that used
        # to collapse into only ``end`` and lose the dictation entirely.
        app._request_capture(True)
        app._request_capture(False)
        app._capture_q.put(None)
        worker = threading.Thread(target=app._capture_worker, daemon=True)
        worker.start()
        worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(operations, ["begin", "end"])

    def test_begin_and_end_are_serialized_off_the_requesting_thread(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._capture_q = queue.SimpleQueue()
        app._capture_requested = False
        app._capture_generation = 0
        app._locked = False

        begin_entered = threading.Event()
        allow_begin_to_finish = threading.Event()
        end_finished = threading.Event()
        operations = []

        def begin(**_kwargs):
            operations.append(("begin-enter", threading.current_thread().name))
            begin_entered.set()
            self.assertTrue(allow_begin_to_finish.wait(2.0))
            operations.append(("begin-exit", threading.current_thread().name))

        def end(**_kwargs):
            operations.append(("end", threading.current_thread().name))
            end_finished.set()

        app._begin = begin
        app._end = end
        worker = threading.Thread(
            target=app._capture_worker, name="test-capture-worker", daemon=True)
        worker.start()

        try:
            requester = threading.current_thread().name
            app._request_capture(True)
            self.assertTrue(begin_entered.wait(2.0))

            # Queue stop while begin is still blocked. The request itself must be
            # non-blocking, and stop may only run after begin has returned.
            app._request_capture(False)
            self.assertFalse(end_finished.is_set())
            allow_begin_to_finish.set()
            self.assertTrue(end_finished.wait(2.0))

            self.assertEqual([name for name, _thread in operations], [
                "begin-enter", "begin-exit", "end",
            ])
            self.assertTrue(all(thread == "test-capture-worker"
                                for _name, thread in operations))
            self.assertNotEqual(requester, "test-capture-worker")
        finally:
            allow_begin_to_finish.set()
            app._capture_q.put(None)
            worker.join(2.0)
        self.assertFalse(worker.is_alive())

    def test_slow_coreaudio_open_does_not_hold_event_state_lock(self):
        class Recorder:
            def __init__(self):
                self.recording = False
                self.opened = threading.Event()
                self.release = threading.Event()

            def start(self):
                self.opened.set()
                if not self.release.wait(2.0):
                    raise TimeoutError("test did not release microphone open")
                self.recording = True

            def stop(self):
                self.recording = False
                return None

            def refresh_after_wake(self):
                return False

        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._capture_q = queue.SimpleQueue()
        app._capture_requested = False
        app._capture_generation = 0
        app._locked = False
        app._record_started = 0.0
        app._warned_capture = False
        app._audio_refresh_pending = False
        app._audio_refresh_generation = 0
        app._audio_refresh_timer = None
        app.recorder = Recorder()
        app.cfg = {
            "learn_from_edits": False,
            "play_sounds": False,
            "min_seconds": 0.1,
        }
        app._set_status = lambda *_args: None
        request_returned = threading.Event()

        with mock.patch.object(flow, "play"), redirect_stdout(io.StringIO()):
            worker = threading.Thread(target=app._capture_worker, daemon=True)
            worker.start()
            app._request_capture(True)
            self.assertTrue(app.recorder.opened.wait(2.0))

            requester = threading.Thread(
                target=lambda: (app._request_capture(False),
                                request_returned.set()),
                daemon=True)
            requester.start()
            responsive = request_returned.wait(0.25)
            app.recorder.release.set()
            requester.join(2.0)
            app._capture_q.put(None)
            worker.join(2.0)

        self.assertTrue(responsive)
        self.assertFalse(requester.is_alive())
        self.assertFalse(worker.is_alive())
        self.assertFalse(app.recorder.recording)

    def test_duplicate_requested_state_is_coalesced(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._capture_q = queue.SimpleQueue()
        app._capture_requested = False
        app._capture_generation = 0
        app._locked = True

        app._request_capture(True)
        app._request_capture(True)
        app._request_capture(False)
        app._request_capture(False)

        self.assertEqual(app._capture_q.get_nowait(), (1, True))
        self.assertEqual(app._capture_q.get_nowait(), (2, False))
        with self.assertRaises(queue.Empty):
            app._capture_q.get_nowait()
        self.assertFalse(app._locked)

    def test_failed_old_start_cannot_erase_newer_repress(self):
        class Recorder:
            def __init__(self):
                self.recording = False
                self.starts = 0

            def start(self):
                self.starts += 1
                if self.starts == 1:
                    raise RuntimeError("simulated wake-time device failure")
                self.recording = True

            def stop(self):
                self.recording = False
                return None

        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._capture_q = queue.SimpleQueue()
        app._capture_requested = False
        app._capture_generation = 0
        app._locked = False
        app._record_started = 0.0
        app._warned_capture = False
        app._audio_refresh_pending = False
        app._audio_refresh_generation = 0
        app._audio_refresh_timer = None
        app.recorder = Recorder()
        app.cfg = {"learn_from_edits": False, "play_sounds": False}
        app._set_status = lambda *_args: None

        # The first open blocks/fails after a complete release/re-press has
        # already made generation 3 the desired state.
        app._request_capture(True)
        app._request_capture(False)
        app._request_capture(True)
        app._capture_q.put(None)
        with mock.patch.object(flow, "play"), redirect_stdout(io.StringIO()):
            worker = threading.Thread(target=app._capture_worker, daemon=True)
            worker.start()
            worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(app.recorder.starts, 2)
        self.assertTrue(app.recorder.recording)
        self.assertTrue(app._capture_requested)

        # Because desired state stayed True, the next release is accepted and
        # queues a real stop instead of being incorrectly coalesced.
        app._request_capture(False)
        self.assertEqual(app._capture_q.get_nowait(), (4, False))


class ModelWarmupQueueTests(unittest.TestCase):
    def test_warm_token_never_consumes_audio_backlog_capacity(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app.transcriber = types.SimpleNamespace(warm_up=lambda: None)
        app._wake_warm_event = threading.Event()
        app._wake_warm_token = object()
        app._work_q = queue.Queue(maxsize=5)
        for i in range(4):
            app._work_q.put_nowait((f"audio-{i}", 0.0))

        app._request_model_warmup()

        self.assertTrue(app._wake_warm_event.is_set())
        self.assertEqual(app._work_q.qsize(), 4)

    def test_idle_worker_receives_warm_token(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app.transcriber = types.SimpleNamespace(warm_up=lambda: None)
        app._wake_warm_event = threading.Event()
        app._wake_warm_token = object()
        app._work_q = queue.Queue(maxsize=5)

        app._request_model_warmup()

        self.assertIs(app._work_q.get_nowait(), app._wake_warm_token)


class RecorderRefreshTests(unittest.TestCase):
    class FakeSoundDevice:
        def __init__(self):
            self.events = []

        def _terminate(self):
            self.events.append("terminate")

        def _initialize(self):
            self.events.append("initialize")

    @staticmethod
    def _recorder(sd, lock):
        recorder = flow.Recorder.__new__(flow.Recorder)
        recorder._sd = sd
        recorder._lock = lock
        recorder.recording = False
        recorder._stream = None
        return recorder

    def test_idle_refresh_reinitializes_portaudio_in_order(self):
        sd = self.FakeSoundDevice()
        recorder = self._recorder(sd, threading.Lock())

        self.assertTrue(recorder.refresh_after_wake())
        self.assertEqual(sd.events, ["terminate", "initialize"])

    def test_refresh_waits_for_lock_and_skips_newly_active_capture(self):
        class GateLock:
            def __init__(self):
                self.entered = threading.Event()
                self.release = threading.Event()

            def __enter__(self):
                self.entered.set()
                if not self.release.wait(2.0):
                    raise TimeoutError("test did not release recorder lock")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        sd = self.FakeSoundDevice()
        lock = GateLock()
        recorder = self._recorder(sd, lock)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(recorder.refresh_after_wake()),
            daemon=True)
        worker.start()

        self.assertTrue(lock.entered.wait(2.0))
        self.assertEqual(sd.events, [])
        recorder.recording = True
        lock.release.set()
        worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [False])
        self.assertEqual(sd.events, [])

    def test_app_retains_busy_refresh_and_retries_when_idle(self):
        class Recorder:
            def __init__(self):
                self.recording = True
                self._stream = object()
                self.calls = 0

            def refresh_after_wake(self):
                self.calls += 1
                return not (self.recording or self._stream is not None)

        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._audio_refresh_pending = True
        app._audio_refresh_generation = 2
        app._audio_refresh_timer = None
        app.recorder = Recorder()

        # A canceled timer from generation 1 cannot consume generation 2.
        self.assertFalse(app._refresh_audio_if_pending(generation=1))
        self.assertEqual(app.recorder.calls, 0)
        self.assertTrue(app._audio_refresh_pending)

        self.assertFalse(app._refresh_audio_if_pending(generation=2))
        self.assertTrue(app._audio_refresh_pending)
        app.recorder.recording = False
        app.recorder._stream = None
        self.assertTrue(app._refresh_audio_if_pending())
        self.assertFalse(app._audio_refresh_pending)
        self.assertEqual(app.recorder.calls, 2)


class ThemedLayerRegistryTests(unittest.TestCase):
    class Layer:
        def __init__(self):
            self.colors = []

        def setBackgroundColor_(self, color):
            self.colors.append(("background", color))

        def setBorderColor_(self, color):
            self.colors.append(("border", color))

    class DynamicColor:
        def __init__(self, value):
            self.value = value

        def CGColor(self):
            return self.value

    def setUp(self):
        self.old_registry = list(flow._THEMED_LAYERS)
        flow._THEMED_LAYERS.clear()

    def tearDown(self):
        flow._THEMED_LAYERS[:] = self.old_registry

    def test_paint_prunes_dead_weakrefs_and_deduplicates_live_layer(self):
        dead = self.Layer()
        dead_ref = weakref.ref(dead)
        del dead
        gc.collect()
        self.assertIsNone(dead_ref())

        existing = self.Layer()
        old_color = self.DynamicColor("old")
        flow._THEMED_LAYERS.extend([
            (dead_ref, "setBackgroundColor_", old_color),
            (weakref.ref(existing), "setBorderColor_", old_color),
        ])

        fake_objc = types.ModuleType("objc")
        fake_objc.WeakRef = weakref.ref
        fresh = self.Layer()
        first_color = self.DynamicColor("first")
        second_color = self.DynamicColor("second")

        with mock.patch.dict(sys.modules, {"objc": fake_objc}), \
                mock.patch.object(flow, "_current_appearance", return_value=None):
            flow._paint(fresh, "setBackgroundColor_", first_color)
            flow._paint(fresh, "setBackgroundColor_", second_color)

        registrations = [
            (ref(), setter, color)
            for ref, setter, color in flow._THEMED_LAYERS
        ]
        self.assertNotIn(None, [layer for layer, _setter, _color in registrations])
        self.assertEqual(len(registrations), 2)
        self.assertTrue(any(layer is existing and setter == "setBorderColor_"
                            for layer, setter, _color in registrations))
        fresh_entries = [entry for entry in registrations if entry[0] is fresh]
        self.assertEqual(fresh_entries, [
            (fresh, "setBackgroundColor_", second_color),
        ])
        self.assertEqual(fresh.colors, [
            ("background", "first"),
            ("background", "second"),
        ])


class HuggingFaceCacheTests(unittest.TestCase):
    @staticmethod
    def _fake_hub(cache_dir: Path):
        package = types.ModuleType("huggingface_hub")
        constants = types.ModuleType("huggingface_hub.constants")
        constants.HF_HUB_CACHE = str(cache_dir)
        package.constants = constants
        return {
            "huggingface_hub": package,
            "huggingface_hub.constants": constants,
        }

    @staticmethod
    def _add_weights(snap_dir: Path):
        """Make a fixture snapshot look fully downloaded (nonzero weights)."""
        (snap_dir / "model.safetensors").write_bytes(b"w" * 8)

    def test_main_ref_wins_over_newer_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            repo = cache / "models--acme--speech-model"
            old = repo / "snapshots" / "old-commit"
            new = repo / "snapshots" / "new-commit"
            old.mkdir(parents=True)
            new.mkdir()
            self._add_weights(old)
            self._add_weights(new)
            now = 1_700_000_000
            os.utime(old, (now, now))
            os.utime(new, (now + 10, now + 10))
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text(
                "old-commit\n", encoding="utf-8")

            with mock.patch.dict(sys.modules, self._fake_hub(cache)):
                resolved = flow._hf_cached_snapshot("acme/speech-model")

            self.assertEqual(resolved, old)

    def test_newest_snapshot_is_fallback_when_main_ref_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            repo = cache / "models--acme--speech-model"
            older = repo / "snapshots" / "older"
            newest = repo / "snapshots" / "newest"
            older.mkdir(parents=True)
            newest.mkdir()
            self._add_weights(older)
            self._add_weights(newest)
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text(
                "missing-commit", encoding="utf-8")
            now = 1_700_000_000
            os.utime(older, (now, now))
            os.utime(newest, (now + 10, now + 10))

            with mock.patch.dict(sys.modules, self._fake_hub(cache)):
                resolved = flow._hf_cached_snapshot("acme/speech-model")

            self.assertEqual(resolved, newest)

    def test_missing_repository_is_not_reported_as_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            with mock.patch.dict(sys.modules, self._fake_hub(cache)):
                self.assertIsNone(flow._hf_cached_snapshot("acme/missing"))

    def test_incomplete_snapshot_is_not_reported_as_cached(self):
        """An interrupted first-run download (config.json present, weights
        absent) must NOT be handed to model loaders: returning it would
        permanently suppress the resumable hub download."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            repo = cache / "models--acme--speech-model"
            snap = repo / "snapshots" / "torn"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}", encoding="utf-8")
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("torn", encoding="utf-8")
            with mock.patch.dict(sys.modules, self._fake_hub(cache)):
                self.assertIsNone(flow._hf_cached_snapshot("acme/speech-model"))

    def test_complete_older_snapshot_beats_incomplete_preferred(self):
        """If the ref'd snapshot is torn but an older complete one exists, the
        complete one is used (offline startup still works)."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            repo = cache / "models--acme--speech-model"
            torn = repo / "snapshots" / "torn"
            good = repo / "snapshots" / "good"
            torn.mkdir(parents=True)
            good.mkdir()
            self._add_weights(good)
            now = 1_700_000_000
            os.utime(good, (now, now))
            os.utime(torn, (now + 10, now + 10))
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("torn", encoding="utf-8")
            with mock.patch.dict(sys.modules, self._fake_hub(cache)):
                resolved = flow._hf_cached_snapshot("acme/speech-model")
            self.assertEqual(resolved, good)


class UsageStatsTests(unittest.TestCase):
    """The History home page's "you've saved N hours" total. It must add up the
    right amount AND never reset — it's a lifetime counter persisted to disk."""

    @staticmethod
    def _tmp_stats(tmp):
        """Point the module's config dir + stats file at a throwaway directory, so
        the suite never reads or writes the real ~/.flowdictate/stats.json."""
        d = Path(tmp)
        return mock.patch.multiple(flow, CONFIG_DIR=d, STATS_PATH=d / "stats.json")

    def test_records_and_accumulates_time_saved(self):
        with tempfile.TemporaryDirectory() as tmp, self._tmp_stats(tmp):
            # 2 words @ 40 WPM -> typed 3.0s; spoke 1.0s -> saved 2.0s.
            flow.record_usage_stats("hello world", 1.0)
            s = flow.load_usage_stats()
            self.assertEqual(s["dictations"], 1)
            self.assertEqual(s["words"], 2)
            self.assertAlmostEqual(s["typed_seconds"], 3.0)
            self.assertAlmostEqual(s["spoken_seconds"], 1.0)
            self.assertAlmostEqual(s["saved_seconds"], 2.0)
            # A second dictation ADDS to the running total (4 words -> typed 6.0,
            # spoke 2.0 -> saved 4.0), it does not replace it.
            flow.record_usage_stats("three more words here", 2.0)
            s = flow.load_usage_stats()
            self.assertEqual(s["dictations"], 2)
            self.assertEqual(s["words"], 6)
            self.assertAlmostEqual(s["typed_seconds"], 9.0)
            self.assertAlmostEqual(s["spoken_seconds"], 3.0)
            self.assertAlmostEqual(s["saved_seconds"], 6.0)

    def test_saved_is_never_negative_for_a_slow_dictation(self):
        with tempfile.TemporaryDirectory() as tmp, self._tmp_stats(tmp):
            # 1 word -> typed 1.5s, but you spoke for 30s. Saved clamps to 0 (you
            # didn't LOSE time on the total), yet the word/dictation still counts.
            flow.record_usage_stats("hi", 30.0)
            s = flow.load_usage_stats()
            self.assertEqual(s["dictations"], 1)
            self.assertEqual(s["words"], 1)
            self.assertAlmostEqual(s["saved_seconds"], 0.0)

    def test_persists_across_a_simulated_restart(self):
        # Two independent calls with the same on-disk file == quit + reopen: the
        # second process must SEE and extend the first's total, not start over.
        with tempfile.TemporaryDirectory() as tmp, self._tmp_stats(tmp):
            flow.record_usage_stats("one two three four five", 1.0)
            flow.record_usage_stats("one two three four five", 1.0)
            self.assertEqual(flow.load_usage_stats()["dictations"], 2)
            self.assertEqual(flow.load_usage_stats()["words"], 10)

    def test_transient_read_failure_does_not_reset_the_total(self):
        # The core "it keeps resetting" guard: if the file can't be read at write
        # time, we must SKIP the update, never overwrite the lifetime total with a
        # freshly-zeroed one.
        with tempfile.TemporaryDirectory() as tmp, self._tmp_stats(tmp):
            for _ in range(3):
                flow.record_usage_stats("one two three four five", 1.0)
            before = flow.load_usage_stats()
            self.assertEqual(before["dictations"], 3)
            with mock.patch.object(
                    flow, "_load_usage_stats_for_update",
                    return_value=(flow._clean_usage_stats({}), False)):
                flow.record_usage_stats("brand new words here now", 1.0)
            after = flow.load_usage_stats()
            self.assertEqual(after["dictations"], 3)          # unchanged
            self.assertEqual(after["words"], before["words"])
            self.assertAlmostEqual(after["saved_seconds"],
                                   before["saved_seconds"])

    def test_corrupt_file_is_reported_not_ok(self):
        with tempfile.TemporaryDirectory() as tmp, self._tmp_stats(tmp):
            flow.STATS_PATH.write_text("{ this is not valid json ]")
            _stats, ok = flow._load_usage_stats_for_update()
            self.assertFalse(ok)

    def test_absent_file_is_a_clean_first_run(self):
        with tempfile.TemporaryDirectory() as tmp, self._tmp_stats(tmp):
            stats, ok = flow._load_usage_stats_for_update()
            self.assertTrue(ok)                               # a real first run
            self.assertEqual(stats["dictations"], 0)


class CorrectionsEscapeTests(unittest.TestCase):
    """apply_corrections must insert taught text LITERALLY. As a re.sub
    template, a taught target containing a backslash either raised re.error
    (silently killing every later dictation until corrections.json was
    hand-edited) or injected control characters into the pasted text."""

    def test_backslash_path_is_inserted_literally(self):
        with mock.patch.object(flow, "load_corrections",
                               return_value={"see users": r"C:\Users\me"}):
            out = flow.apply_corrections("please see users now")
        self.assertEqual(out, r"please C:\Users\me now")

    def test_backslash_sequences_are_not_escapes(self):
        # "\a" (BEL) and "\n" (newline) must come out as two characters each.
        with mock.patch.object(flow, "load_corrections",
                               return_value={"alpha": "\\alpha", "newline": "\\n"}):
            out = flow.apply_corrections("alpha then newline")
        self.assertEqual(out, "\\alpha then \\n")


class DefaultConfigWriteTests(unittest.TestCase):
    """--setup runs write_default_config on every install/update; it must never
    clobber an existing, possibly hand-tuned config.json."""

    def test_existing_config_is_left_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "config.json"
            payload = '{"hotkey": "cmd_r", "cleanup": "basic", "custom": 1}\n'
            cfg_path.write_text(payload, encoding="utf-8")
            with mock.patch.object(flow, "CONFIG_DIR", tmp_path), \
                 mock.patch.object(flow, "CONFIG_PATH", cfg_path), \
                 mock.patch.object(flow, "CODE_DIR_PATH", tmp_path / "code_dir"), \
                 redirect_stdout(io.StringIO()):
                flow.write_default_config()
            self.assertEqual(cfg_path.read_text(encoding="utf-8"), payload)
            # the code-dir pointer is the one thing it should (re)write
            self.assertTrue((tmp_path / "code_dir").is_file())


if __name__ == "__main__":
    unittest.main()


class TapKindTests(unittest.TestCase):
    def test_modifier_hotkeys_use_a_passive_listener(self):
        self.assertTrue(flow._hotkey_is_modifier_only({58}))
        self.assertTrue(flow._hotkey_is_modifier_only({58, 61}))
        self.assertTrue(flow._hotkey_is_modifier_only(
            flow.resolve_target_vks("alt_l")))
        self.assertTrue(flow._hotkey_is_modifier_only(
            flow.resolve_target_vks("vk:63")))          # Fn / Globe

    def test_regular_key_hotkeys_need_an_active_tap(self):
        self.assertFalse(flow._hotkey_is_modifier_only(
            flow.resolve_target_vks("vk:96")))          # F5
        self.assertFalse(flow._hotkey_is_modifier_only({58, 96}))
        self.assertFalse(flow._hotkey_is_modifier_only(set()))


class ReleaseWatchTests(unittest.TestCase):
    """A hold-mode key-up that never reaches the event tap (macOS disabled the
    tap for a moment) must not keep the microphone open until the 120 s cap."""

    @staticmethod
    def _readings(*values):
        """A physical-key fake replaying `values`, then repeating the last."""
        seen = []

        def physically_down():
            seen.append(1)
            return values[min(len(seen), len(values)) - 1]

        physically_down.seen = seen
        return physically_down

    @staticmethod
    def _app(*, physically_down, locked=False, mode="hold"):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._capture_q = queue.SimpleQueue()
        app._capture_requested = True
        app._capture_generation = 1
        app._locked = locked
        app._key_down = True
        app._keystate_confirmed = False
        app._keystate_warned = False
        app._record_started = time.monotonic() - 5.0    # held well past settling
        app.recorder = types.SimpleNamespace(recording=True)
        app.cfg = {"mode": mode}
        app.target_vks = {58}
        app._hotkey_physically_down = physically_down
        app._RELEASE_POLL_SECONDS = 0.005
        return app

    def test_lost_key_up_is_recovered_from_the_physical_key_state(self):
        # The window server confirms the key down, then reads it up for good.
        down = self._readings(True, False)
        app = self._app(physically_down=down)
        with redirect_stdout(io.StringIO()) as out:
            app._release_watch_loop(1)
        self.assertEqual(app._capture_q.get_nowait(), (2, False))
        self.assertFalse(app._capture_requested)
        self.assertFalse(app._key_down)
        self.assertEqual(len(down.seen), 4)      # 1 confirm + 3 consecutive "up"
        self.assertIn("release watcher armed", out.getvalue())
        self.assertIn("never reached the event tap", out.getvalue())

    def test_needs_consecutive_up_readings(self):
        # A single flicker between "down" readings must never end a capture.
        down = self._readings(False, True, False, False, True, False, False, False)
        app = self._app(physically_down=down)
        with redirect_stdout(io.StringIO()):
            app._release_watch_loop(1)
        self.assertEqual(len(down.seen), 8)
        self.assertEqual(app._capture_q.get_nowait(), (2, False))

    def test_a_key_never_seen_down_is_never_acted_on(self):
        # A keyboard the window server cannot see: the watcher must stay quiet
        # until the capture ends for a real reason, and say so once.
        down = self._readings(False)
        app = self._app(physically_down=down)

        def real_stop():
            time.sleep(0.05)
            with app._state_lock:
                app.recorder.recording = False

        threading.Thread(target=real_stop, daemon=True).start()
        with redirect_stdout(io.StringIO()) as out:
            app._release_watch_loop(1)
        self.assertGreater(len(down.seen), 2)
        self.assertTrue(app._capture_requested)
        with self.assertRaises(queue.Empty):
            app._capture_q.get_nowait()
        self.assertIn("never saw the hotkey held", out.getvalue())
        self.assertTrue(app._keystate_warned)

    def test_a_held_key_never_stops_the_capture(self):
        reads = []

        def physically_down():
            reads.append(1)
            if len(reads) >= 6:
                with app._state_lock:
                    app._capture_generation = 2   # the tap delivered a real release
            return True

        app = self._app(physically_down=physically_down)
        app._release_watch_loop(1)
        self.assertTrue(app._capture_requested)
        with self.assertRaises(queue.Empty):
            app._capture_q.get_nowait()

    def test_locked_capture_ignores_a_physical_release(self):
        app = self._app(physically_down=lambda: False, locked=True)
        app._release_watch_loop(1)     # hands-free latch: returns at once
        self.assertTrue(app._capture_requested)
        with self.assertRaises(queue.Empty):
            app._capture_q.get_nowait()

    def test_fresh_press_is_left_to_the_tap_while_it_settles(self):
        # Confirmed down, then "up" readings — but inside the settle window a
        # real release (generation bump) must be the only thing that ends it.
        app = self._app(physically_down=self._readings(True, False))
        app._record_started = time.monotonic()      # pressed just now

        def real_release():
            time.sleep(0.05)
            with app._state_lock:
                app._capture_generation = 2

        threading.Thread(target=real_release, daemon=True).start()
        app._release_watch_loop(1)
        self.assertTrue(app._capture_requested)
        with self.assertRaises(queue.Empty):
            app._capture_q.get_nowait()

    def test_toggle_mode_never_starts_a_watcher(self):
        app = self._app(physically_down=lambda: False, mode="toggle")
        started = []
        with mock.patch.object(threading, "Thread",
                               side_effect=lambda *a, **k: started.append(k)):
            app._start_release_watch(1)
        self.assertEqual(started, [])


class KeyStateProbeTests(unittest.TestCase):
    def test_physical_state_read_is_conservative(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app.target_vks = {58}
        fake = types.SimpleNamespace(
            kCGEventSourceStateCombinedSessionState=0,
            CGEventSourceKeyState=lambda _state, _vk: False,
            CGEventSourceFlagsState=lambda _state: 0x80000)
        with mock.patch.dict(flow._MODIFIER_CLASS_MASK_BY_VK, {58: 0x80000}), \
                mock.patch.dict(sys.modules, {"Quartz": fake}):
            # Key up, but the Option class flag is still set (sibling key held).
            self.assertTrue(app._hotkey_physically_down())
            fake.CGEventSourceFlagsState = lambda _state: 0x100
            self.assertFalse(app._hotkey_physically_down())

            def boom(_state, _vk):
                raise RuntimeError("window server unavailable")

            fake.CGEventSourceKeyState = boom
            # A failed read can never cut a live dictation short.
            self.assertTrue(app._hotkey_physically_down())


class ForcedStopTests(unittest.TestCase):
    def test_runaway_guard_can_close_a_recorder_that_outlived_its_request(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._capture_q = queue.SimpleQueue()
        app._capture_requested = False      # desired state already says "stopped"
        app._capture_generation = 3
        app._locked = True
        app.recorder = types.SimpleNamespace(recording=True)

        self.assertFalse(app._request_capture(False))               # coalesced
        self.assertTrue(app._request_capture(False, force=True))    # still queued
        self.assertEqual(app._capture_q.get_nowait(), (4, False))
        self.assertFalse(app._locked)

        # force never invents a stop for a recorder that is already closed.
        app.recorder.recording = False
        self.assertFalse(app._request_capture(False, force=True))
        with self.assertRaises(queue.Empty):
            app._capture_q.get_nowait()


class PostWakeWarmupTests(unittest.TestCase):
    @staticmethod
    def _app():
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._wake_warm_event = threading.Event()
        app._wake_warm_token = object()
        app._work_q = queue.Queue(maxsize=5)
        app._capture_requested = False
        app.recorder = types.SimpleNamespace(recording=False)
        app._processing_started = None
        app._proc_warned = False
        app.warmed = 0
        app.processed = []

        def warm():
            app.warmed += 1
            app._wake_warm_event.clear()

        app._warm_models_after_wake = warm
        app._process = lambda audio, *, queue_wait=0.0: app.processed.append(audio)
        app._reflect_pipeline_status = lambda: None
        return app

    def _run_worker(self, app):
        worker = threading.Thread(target=app._transcription_worker, daemon=True)
        worker.start()
        # Let the worker drain what was queued BEFORE handing it the shutdown
        # sentinel; a sentinel already in the queue would look like a waiting
        # clip to the warm-up decision.
        deadline = time.monotonic() + 2.0
        while app._work_q.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(app._work_q.unfinished_tasks, 0)
        app._work_q.put_nowait(None)
        worker.join(2.0)
        self.assertFalse(worker.is_alive())

    def test_idle_wake_still_warms_the_model(self):
        app = self._app()
        app._wake_warm_event.set()
        app._work_q.put_nowait(app._wake_warm_token)
        self._run_worker(app)
        self.assertEqual(app.warmed, 1)

    def test_warm_up_never_runs_ahead_of_a_waiting_clip(self):
        app = self._app()
        app._wake_warm_event.set()
        app._work_q.put_nowait(app._wake_warm_token)
        app._work_q.put_nowait(("clip", 0.0))
        self._run_worker(app)
        self.assertEqual(app.warmed, 0)
        self.assertEqual(app.processed, ["clip"])
        self.assertFalse(app._wake_warm_event.is_set())

    def test_warm_up_yields_to_an_open_microphone(self):
        app = self._app()
        app._wake_warm_event.set()
        app.recorder.recording = True
        app._work_q.put_nowait(app._wake_warm_token)
        self._run_worker(app)
        self.assertEqual(app.warmed, 0)
        self.assertFalse(app._wake_warm_event.is_set())

    def test_real_clip_clears_a_pending_warm_up_instead_of_paying_twice(self):
        app = self._app()
        app._wake_warm_event.set()     # token could not be queued: queue was full
        app._work_q.put_nowait(("clip", 0.0))
        self._run_worker(app)
        self.assertEqual(app.warmed, 0)
        self.assertEqual(app.processed, ["clip"])
        self.assertFalse(app._wake_warm_event.is_set())


class CaptureStopOrderTests(unittest.TestCase):
    @staticmethod
    def _app(audio):
        class Recorder:
            recording = True

            def stop(self):
                self.recording = False
                return audio

        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._capture_requested = True
        app._capture_generation = 1
        app._locked = True
        app._record_started = 1.0
        app._processing_started = None
        app._work_q = queue.Queue(maxsize=5)
        app.cfg = {"min_seconds": 0.1, "play_sounds": False}
        app.recorder = Recorder()
        app.events = []
        app._set_status = lambda glyph, _label: app.events.append(("status", glyph))
        app._refresh_audio_if_pending = (
            lambda *_a, **_k: app.events.append(("refresh",)) or False)
        return app

    def test_status_is_published_before_the_post_wake_audio_refresh(self):
        clip = flow.np.zeros(flow.SAMPLE_RATE, dtype=flow.np.float32)
        app = self._app(clip)
        with mock.patch.object(flow, "play"), redirect_stdout(io.StringIO()):
            app._end(generation=1)
        self.assertEqual(app.events, [("status", "⏳"), ("refresh",)])
        self.assertEqual(app._work_q.qsize(), 1)
        self.assertFalse(app._locked)
        self.assertFalse(app._capture_requested)

    def test_too_short_clip_still_refreshes_after_going_idle(self):
        app = self._app(None)
        with mock.patch.object(flow, "play"), redirect_stdout(io.StringIO()):
            app._end(generation=1)
        self.assertEqual(app.events, [("status", "🎙️"), ("refresh",)])
        self.assertTrue(app._work_q.empty())

    def test_status_is_published_while_holding_the_state_lock(self):
        app = self._app(None)
        owned = []
        app._set_status = lambda _g, _l: owned.append(app._state_lock._is_owned())
        app._reflect_pipeline_status()
        self.assertEqual(owned, [True])
        self.assertFalse(app._state_lock._is_owned())


class EditLearningTests(unittest.TestCase):
    def test_diff_runs_off_the_calling_thread_and_only_once(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._pending_learn = {"el": "field", "pasted": "hello",
                              "done": False, "generation": 1}
        app._learn_timer = None
        seen = []
        done = threading.Event()

        def fake_learn(el, pasted, **_kwargs):
            seen.append((el, pasted, threading.current_thread().name))
            done.set()
            return 0

        with mock.patch.object(flow, "learn_from_edit", side_effect=fake_learn):
            app._reconcile_edit_learning()
            app._reconcile_edit_learning()      # nothing pending any more
            self.assertTrue(done.wait(2.0))
            time.sleep(0.05)
        self.assertEqual([s[:2] for s in seen], [("field", "hello")])
        self.assertNotEqual(seen[0][2], threading.current_thread().name)
        self.assertIsNone(app._pending_learn)


class RecorderStopTests(unittest.TestCase):
    def test_slow_stream_close_is_reported(self):
        class Stream:
            def stop(self):
                pass

            def close(self):
                pass

        rec = flow.Recorder.__new__(flow.Recorder)
        rec._lock = threading.Lock()
        rec._frames_lock = threading.Lock()
        rec._frames = [flow.np.zeros((160, 1), dtype=flow.np.float32)]
        rec._stream = Stream()
        rec.recording = True
        with mock.patch.object(flow.time, "monotonic", side_effect=[0.0, 2.5]), \
                redirect_stdout(io.StringIO()) as out:
            audio = rec.stop()
        self.assertIn("took 2.5s to close", out.getvalue())
        self.assertEqual(audio.shape, (160,))
        self.assertFalse(rec.recording)
        self.assertIsNone(rec._stream)


class LogTimestampTests(unittest.TestCase):
    def test_lines_get_a_stamp_and_partial_writes_do_not(self):
        raw = io.StringIO()
        stream = flow._TimestampedStream(raw)
        with mock.patch.object(flow.time, "strftime", return_value="[T] "):
            stream.write("[flow] one\n")
            stream.write("partial ")
            stream.write("rest\n")
            stream.write("\n")
            stream.write("\rprogress 50%")
            stream.write("\rprogress 100%\n")
            stream.write("after\n")
        self.assertEqual(
            raw.getvalue(),
            "[T] [flow] one\n[T] partial rest\n\n"
            "\rprogress 50%\rprogress 100%\n[T] after\n")

    def test_terminal_output_is_left_alone(self):
        tty = io.StringIO()
        tty.isatty = lambda: True
        with mock.patch.object(sys, "stdout", tty), \
                mock.patch.object(sys, "stderr", tty):
            flow._timestamp_log_streams()
            self.assertIs(sys.stdout, tty)

    def test_redirected_output_is_wrapped_exactly_once(self):
        log = io.StringIO()         # isatty() is False
        with mock.patch.object(sys, "stdout", log), \
                mock.patch.object(sys, "stderr", log):
            flow._timestamp_log_streams()
            wrapped = sys.stdout
            self.assertIsInstance(wrapped, flow._TimestampedStream)
            flow._timestamp_log_streams()
            self.assertIs(sys.stdout, wrapped)
            # Everything else (fileno, encoding, ...) is delegated to the raw
            # stream, so libraries that poke at sys.stdout keep working.
            with mock.patch.object(flow.time, "strftime", return_value="[T] "):
                sys.stdout.write("hello\n")
            self.assertEqual(log.getvalue(), "[T] hello\n")
            self.assertFalse(sys.stdout.isatty())
