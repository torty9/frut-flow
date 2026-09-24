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
import signal
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


class RelativeTimeTests(unittest.TestCase):
    """The History meta line's "when": short, and worded the way macOS does."""

    # Local noon in mid-January: day buckets are calendar-based, so anchoring to
    # local time (far from any DST change) keeps this true in every timezone.
    NOW = time.mktime((2027, 1, 15, 12, 0, 0, 0, 0, -1))

    def ago(self, seconds):
        return flow.relative_time(self.NOW - seconds, now=self.NOW)

    def test_minutes_and_hours_are_spelled_short_not_abbreviated(self):
        self.assertEqual(self.ago(10), "Just now")
        self.assertEqual(self.ago(2 * 60), "2 min ago")
        self.assertEqual(self.ago(41 * 60), "41 min ago")
        self.assertEqual(self.ago(3 * 3600 + 5), "3 hours ago")

    def test_singular_hour_and_the_sixty_minute_edge(self):
        self.assertEqual(self.ago(3600 + 30), "1 hour ago")
        # 59.5–60 min rounds to 60: that is an hour, never "60 min ago".
        self.assertEqual(self.ago(59 * 60 + 50), "1 hour ago")

    def test_future_and_garbage_timestamps_never_raise(self):
        self.assertEqual(self.ago(-500), "Just now")
        self.assertEqual(flow.relative_time("not a time", now=self.NOW), "")
        self.assertEqual(flow.relative_time(None, now=self.NOW), "")

    def test_older_entries_bucket_by_calendar_day(self):
        self.assertEqual(self.ago(30 * 3600), "Yesterday")
        self.assertEqual(self.ago(4 * 86400), "4 days ago")
        self.assertEqual(self.ago(30 * 86400), "Dec 16")


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


class DisplayWakeTests(unittest.TestCase):
    """A display wake with no system sleep behind it (screen saver, idle display
    sleep) must not rebuild the taps or warm the models; a real wake must."""

    @staticmethod
    def _app():
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app._system_slept = False
        app._wake_recovery_pending = False
        app.full = []
        app.light = []
        app._recover_after_wake = (
            lambda reason, **kw: app.full.append((reason, kw)) or True)
        app._after_display_only_wake = lambda: app.light.append("light")
        return app

    def test_display_only_wake_takes_the_light_path(self):
        app = self._app()
        app._on_screens_did_wake()
        self.assertEqual(app.full, [])
        self.assertEqual(app.light, ["light"])

    def test_display_wake_after_system_wake_recovers_fully(self):
        app = self._app()
        app._on_system_did_wake()
        self.assertTrue(app._system_slept)
        app._on_screens_did_wake()
        self.assertEqual([r for r, _kw in app.full],
                         ["mac woke from sleep", "displays woke (lid open?)"])
        self.assertEqual(app.full[1][1], {"visible": True})
        self.assertEqual(app.light, [])

    def test_deferred_dark_wake_still_recovers_on_display_wake(self):
        app = self._app()
        app._wake_recovery_pending = True     # DidWake arrived while dark
        app._on_screens_did_wake()
        self.assertEqual(len(app.full), 1)
        self.assertEqual(app.light, [])

    def test_full_runtime_recovery_clears_the_slept_flag(self):
        app = WakeRecoveryTests._app()
        app._system_slept = True
        with mock.patch.object(flow.time, "time", return_value=100.0):
            self.assertTrue(app._recover_after_wake(
                "display woke", visible=True, runtime=True))
        self.assertFalse(app._system_slept)

    def test_light_path_reenables_a_disabled_tap_only(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._tap, app._lock_tap = object(), object()
        enabled = {app._tap: False, app._lock_tap: True}
        fake_quartz = types.SimpleNamespace(
            CGEventTapIsEnabled=lambda t: enabled[t],
            CGEventTapEnable=lambda t, on: enabled.__setitem__(t, on))
        with mock.patch.dict(sys.modules, {"Quartz": fake_quartz}), \
                redirect_stdout(io.StringIO()) as out:
            app._after_display_only_wake()
        self.assertTrue(enabled[app._tap])
        self.assertIn("re-enabled the hotkey tap", out.getvalue())


class RebindHotkeyTests(unittest.TestCase):
    def _app(self, mode):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = threading.RLock()
        app.cfg = {"mode": mode, "hotkey": "alt_r"}
        app.hotkey_name = "alt_r"
        app.target_vks = {61}
        app._key_down = True
        app._locked = False
        app._last_toggle = 0.0
        app.recorder = types.SimpleNamespace(recording=False)
        app._capture_requested = True      # the captured press started a capture
        app.stops = []
        app._request_capture = lambda begin: app.stops.append(begin)
        app._invalidate_popover = lambda: None
        return app

    def test_rebind_in_toggle_mode_stops_the_capture_the_press_started(self):
        app = self._app("toggle")
        with redirect_stdout(io.StringIO()):
            self.assertTrue(app.apply_hotkey("vk:49"))
        self.assertEqual(app.stops, [False])
        self.assertEqual(app.target_vks, {49})
        self.assertFalse(app._key_down)

    def test_rebind_in_hold_mode_still_stops_it(self):
        app = self._app("hold")
        with redirect_stdout(io.StringIO()):
            self.assertTrue(app.apply_hotkey("cmd_r"))
        self.assertEqual(app.stops, [False])


class HotkeyCaptureGuardTests(unittest.TestCase):
    """The Settings capture must refuse keys you type with: a non-modifier
    hotkey is consumed system-wide, so Space/Return/a letter would stop typing
    in every app."""

    def test_typing_keys_are_refused(self):
        for vk, label in ((49, "Space"), (36, "Return"), (48, "Tab"),
                          (51, "Delete"), (0, "A"), (18, "1"), (50, "`"),
                          (76, "Enter"), (117, "Forward Delete"),
                          (123, "Left Arrow")):
            problem = flow._hotkey_capture_problem(vk)
            self.assertIsNotNone(problem, f"vk {vk} ({label}) should be refused")
            self.assertTrue(problem.startswith(label), problem)

    def test_modifiers_function_keys_and_keypad_are_fine(self):
        for vk in (61, 58, 54, 55, 59, 62, 56, 60,     # modifiers, both sides
                   57, 63,                             # Caps Lock, Fn/Globe
                   122, 120, 105, 107, 113,            # F1, F2, F13, F14, F15
                   82, 65, 67,                         # keypad 0 . *
                   10,                                 # ISO section key
                   115, 119, 116, 121):                # Home End PgUp PgDn
            self.assertIsNone(flow._hotkey_capture_problem(vk), f"vk {vk}")

    def test_config_file_still_accepts_any_key(self):
        # The guard is a Settings-UI courtesy, not a config restriction.
        self.assertTrue(flow._valid_hotkey("vk:49"))
        self.assertEqual(flow._normalize_config({"hotkey": "vk:49"})["hotkey"],
                         "vk:49")


class RecorderTimingTests(unittest.TestCase):
    def test_open_and_start_are_timed_separately(self):
        rec = flow.Recorder.__new__(flow.Recorder)
        rec.sample_rate = flow.SAMPLE_RATE
        rec._callback = lambda *a: None
        started = []

        class FakeStream:
            def start(self):
                started.append(True)

        rec._sd = types.SimpleNamespace(InputStream=lambda **kw: FakeStream())
        clock = iter([10.000, 10.060, 10.085])
        with mock.patch.object(flow.time, "monotonic", side_effect=lambda: next(clock)):
            stream = rec._open_stream()
        self.assertIsInstance(stream, FakeStream)
        self.assertEqual(started, [True])
        self.assertAlmostEqual(rec.last_open_ms, 60.0, places=3)
        self.assertAlmostEqual(rec.last_start_ms, 25.0, places=3)


class ClipboardRestoreTests(unittest.TestCase):
    """Two dictations inside the 2 s restore window must give the user back
    their ORIGINAL clipboard — not the first dictation's text."""

    class FakePasteboard:
        def __init__(self, initial):
            self.text, self.count = initial, 1

        def set(self, text):
            self.text, self.count = text, self.count + 1
            return self.count

        def snapshot(self):
            return ("text", self.text) if self.text else None

        def restore(self, snap):
            return self.set(snap[1])

    class FakeTimer:
        pending = []

        def __init__(self, delay, fn):
            self.delay, self.fn, self.cancelled = delay, fn, False
            ClipboardRestoreTests.FakeTimer.pending.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    def _patched(self, pb):
        self.FakeTimer.pending.clear()
        flow._CLIP_PENDING = None
        self.addCleanup(setattr, flow, "_CLIP_PENDING", None)
        stack = [
            mock.patch.object(flow, "_ax_trusted", return_value=True),
            mock.patch.object(flow, "_secure_input_active", return_value=False),
            mock.patch.object(flow, "_send_paste"),
            mock.patch.object(flow, "_clip_set", side_effect=pb.set),
            mock.patch.object(flow, "_clip_snapshot", side_effect=pb.snapshot),
            mock.patch.object(flow, "_clip_change_count",
                              side_effect=lambda: pb.count),
            mock.patch.object(flow, "_clip_restore", side_effect=pb.restore),
            mock.patch.object(flow.threading, "Timer", self.FakeTimer),
        ]
        for p in stack:
            p.start()
            self.addCleanup(p.stop)

    CFG = {"insert_method": "paste", "restore_clipboard": True}

    def test_single_paste_restores_the_original(self):
        pb = self.FakePasteboard("ORIGINAL")
        self._patched(pb)
        self.assertTrue(flow.insert_text(" first", self.CFG))
        self.assertEqual(pb.text, " first")
        self.FakeTimer.pending[0].fn()
        self.assertEqual(pb.text, "ORIGINAL")
        self.assertIsNone(flow._CLIP_PENDING)

    def test_rapid_second_dictation_keeps_the_original_clipboard(self):
        pb = self.FakePasteboard("ORIGINAL")
        self._patched(pb)
        flow.insert_text(" first", self.CFG)
        flow.insert_text(" second", self.CFG)   # before the first restore fired
        timers = list(self.FakeTimer.pending)
        self.assertEqual(len(timers), 2)
        self.assertTrue(timers[0].cancelled)
        self.assertEqual(pb.text, " second")
        timers[1].fn()
        self.assertEqual(pb.text, "ORIGINAL")   # not " first"

    def test_restore_is_skipped_when_someone_else_copied_meanwhile(self):
        pb = self.FakePasteboard("ORIGINAL")
        self._patched(pb)
        flow.insert_text(" first", self.CFG)
        pb.set("USER COPIED THIS")
        self.FakeTimer.pending[0].fn()
        self.assertEqual(pb.text, "USER COPIED THIS")

    def test_second_paste_after_a_user_copy_snapshots_the_new_clipboard(self):
        pb = self.FakePasteboard("ORIGINAL")
        self._patched(pb)
        flow.insert_text(" first", self.CFG)
        pb.set("USER COPIED THIS")
        flow.insert_text(" second", self.CFG)
        timers = list(self.FakeTimer.pending)
        self.assertFalse(timers[0].cancelled)   # not ours any more: left alone
        timers[0].fn()                          # count moved on: no-op
        timers[1].fn()
        self.assertEqual(pb.text, "USER COPIED THIS")


class RepairPromptCacheTests(unittest.TestCase):
    """_LocalRepairer.generate_chat must feed the model only the tokens that
    follow the cached prefix, keep exactly the prefix resident afterwards, and
    fall back to a plain uncached call on any surprise."""

    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=False,
                                tokenize=False):
            toks = []
            for m in messages:
                base = (hash((m["role"], m["content"])) & 0xFFF) + 1
                toks += [base, base + 1, base + 2]
            if add_generation_prompt:
                toks.append(7)
            return toks if tokenize else "T:" + ",".join(map(str, toks))

    class FakeKV:
        def __init__(self):
            self.offset = 0

        def is_trimmable(self):
            return True

        def trim(self, n):
            n = min(self.offset, n)
            self.offset -= n
            return n

        def __len__(self):
            return self.offset

        @property
        def state(self):
            return []

    class FakeCacheMod:
        def __init__(self):
            self.made = 0

        def make_prompt_cache(self, model):
            self.made += 1
            return [RepairPromptCacheTests.FakeKV()]

        def can_trim_prompt_cache(self, cache):
            return True

        def trim_prompt_cache(self, cache, n):
            return [c.trim(n) for c in cache][0]

        def cache_length(self, cache):
            return max(len(c) for c in cache)

    def _repairer(self):
        rep = flow._LocalRepairer.__new__(flow._LocalRepairer)
        rep.repo_id = "fake"
        rep.model = object()
        rep.tokenizer = self.FakeTokenizer()
        rep._make_sampler = lambda temp: None
        rep._cache_mod = self.FakeCacheMod()
        rep._cache = None
        rep._cache_tokens = []
        rep.calls = []

        def fake_generate(model, tokenizer, prompt, *, max_tokens, sampler=None,
                          verbose=False, prompt_cache=None):
            rep.calls.append((prompt, prompt_cache is not None))
            if prompt_cache is not None:
                prompt_cache[0].offset += len(prompt) + 5   # prompt + answer
            return "fixed"
        rep._generate = fake_generate
        fake_mx = types.SimpleNamespace(eval=lambda *a, **k: None)
        patcher = mock.patch.dict(sys.modules, {"mlx": types.SimpleNamespace(core=fake_mx),
                                                "mlx.core": fake_mx})
        patcher.start()
        self.addCleanup(patcher.stop)
        return rep

    @staticmethod
    def _messages(system, user):
        return ([{"role": "system", "content": system}]
                + [{"role": "user", "content": "peer"},
                   {"role": "assistant", "content": "pier"}]
                + [{"role": "user", "content": user}])

    def test_same_prefix_only_feeds_the_new_user_turn(self):
        rep = self._repairer()
        tok = rep.tokenizer
        m1 = self._messages("sys A", "hello there")
        m2 = self._messages("sys A", "another sentence")
        self.assertEqual(rep.generate_chat(m1, max_tokens=8, temp=0.0), "fixed")
        full1 = tok.apply_chat_template(m1, add_generation_prompt=True, tokenize=True)
        prefix = tok.apply_chat_template(m1[:-1], tokenize=True)
        self.assertEqual(rep.calls[0], (full1, True))       # cold: everything
        self.assertEqual(rep._cache_tokens, prefix)
        self.assertEqual(len(rep._cache[0]), len(prefix))   # answer trimmed off

        rep.generate_chat(m2, max_tokens=8, temp=0.0)
        full2 = tok.apply_chat_template(m2, add_generation_prompt=True, tokenize=True)
        self.assertEqual(rep.calls[1], (full2[len(prefix):], True))
        self.assertEqual(len(rep._cache[0]), len(prefix))
        self.assertEqual(rep._cache_mod.made, 1)

    def test_changed_system_prompt_refills_from_the_common_prefix(self):
        rep = self._repairer()
        rep.generate_chat(self._messages("sys A", "one"), max_tokens=8, temp=0.0)
        m3 = self._messages("sys B", "two")
        rep.generate_chat(m3, max_tokens=8, temp=0.0)
        full3 = rep.tokenizer.apply_chat_template(
            m3, add_generation_prompt=True, tokenize=True)
        self.assertEqual(rep.calls[1], (full3, True))       # nothing shared: refill
        prefix3 = rep.tokenizer.apply_chat_template(m3[:-1], tokenize=True)
        self.assertEqual(rep._cache_tokens, prefix3)

    def test_without_cache_module_plain_string_prompt_is_used(self):
        rep = self._repairer()
        rep._cache_mod = None
        m = self._messages("sys A", "one")
        rep.generate_chat(m, max_tokens=8, temp=0.0)
        expected = rep.tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False)
        self.assertEqual(rep.calls, [(expected, False)])

    def test_cache_failure_falls_back_and_resets(self):
        rep = self._repairer()
        rep.generate_chat(self._messages("sys A", "one"), max_tokens=8, temp=0.0)

        def boom(cache, n):
            raise RuntimeError("cache api drift")
        rep._cache_mod.trim_prompt_cache = boom
        m = self._messages("sys A", "two")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rep.generate_chat(m, max_tokens=8, temp=0.0), "fixed")
        self.assertEqual(rep.calls[-1][1], False)            # uncached
        self.assertIsInstance(rep.calls[-1][0], str)
        self.assertIsNone(rep._cache)
        self.assertEqual(rep._cache_tokens, [])


class HistoryLoadTests(unittest.TestCase):
    def test_malformed_numeric_fields_are_coerced(self):
        entries = [{"text": "hello there", "ts": "soon", "words": "twelve"},
                   {"text": "ok", "ts": 1.5, "words": 1, "delivered": 0}]
        with mock.patch.object(flow, "_read_json", return_value=entries):
            hist = flow.load_history()
        self.assertEqual(hist[1]["words"], 2)       # recomputed from the text
        self.assertEqual(hist[1]["ts"], 0.0)
        self.assertEqual(hist[0]["words"], 1)
        self.assertFalse(hist[0]["delivered"])


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


class MenuBarChildProcessTests(unittest.TestCase):
    """On macOS 27 a bundle launch must run the app in a child process, or the
    menu-bar 🎙️ is never placed (see the top of flow.py)."""

    ARGV = ["/Apps/frutflow.app/Contents/MacOS/python3", "/code/flow.py", "--app"]

    @staticmethod
    def _close_quietly(*fds):
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def _launch(self, env=None, *, major=27, argv=None, exit_code=0):
        """Call _run_app_in_child_if_needed as a LaunchServices launch would.
        Returns (Popen mock, child mock, installed handlers, exit code, pipe)."""
        env = {"__CFBundleIdentifier": flow.APP_BUNDLE_ID} if env is None else env
        child = mock.Mock(pid=4242)
        child.wait.return_value = exit_code
        handlers = {}
        r, w = os.pipe()
        self.addCleanup(self._close_quietly, r, w)
        code = None
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(flow.sys, "platform", "darwin"), \
                mock.patch.object(flow.sys, "argv", argv or self.ARGV[1:]), \
                mock.patch.object(flow.sys, "orig_argv", self.ARGV), \
                mock.patch.object(flow.sys, "executable", self.ARGV[0]), \
                mock.patch.object(flow, "_macos_major", return_value=major), \
                mock.patch.object(flow.os, "pipe", return_value=(r, w)), \
                mock.patch.object(flow.subprocess, "Popen",
                                  return_value=child) as popen, \
                mock.patch.object(flow.signal, "signal",
                                  side_effect=handlers.__setitem__), \
                redirect_stdout(io.StringIO()):
            if "__CFBundleIdentifier" not in env:
                os.environ.pop("__CFBundleIdentifier", None)
            os.environ.pop(flow._APP_CHILD_ENV, None)
            try:
                flow._run_app_in_child_if_needed()
            except SystemExit as e:
                code = e.code
        return popen, child, handlers, code, (r, w)

    def test_bundle_launch_runs_the_app_as_a_child_and_mirrors_its_exit(self):
        popen, _, _, code, (r, w) = self._launch(exit_code=3)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], self.ARGV)      # same interpreter, script, flags
        self.assertEqual(kwargs["pass_fds"], (r,))
        self.assertEqual(kwargs["env"][flow._APP_CHILD_ENV], str(r))
        self.assertEqual(code, 3)
        with self.assertRaises(OSError):          # the parent keeps only the write end
            os.fstat(r)
        os.fstat(w)

    def test_signals_to_the_parent_are_forwarded_to_the_child(self):
        _, child, handlers, _, _ = self._launch()
        self.assertEqual(set(handlers),
                         {signal.SIGTERM, signal.SIGINT, signal.SIGHUP})
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        child.send_signal.assert_called_once_with(signal.SIGTERM)

    def test_child_killed_by_a_signal_exits_like_a_shell_reports_it(self):
        *_, code, _ = self._launch(exit_code=-signal.SIGTERM)
        self.assertEqual(code, 128 + signal.SIGTERM)

    def test_other_launches_keep_the_app_in_this_process(self):
        cases = {
            "not launched by LaunchServices": dict(env={}),
            "terminal": dict(env={"__CFBundleIdentifier": "com.apple.Terminal"}),
            "--no-menubar": dict(argv=["/code/flow.py", "--no-menubar"]),
            "macOS 26": dict(major=26),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                popen, _, handlers, code, _ = self._launch(**kwargs)
                popen.assert_not_called()
                self.assertEqual(handlers, {})
                self.assertIsNone(code)

    def test_the_child_watches_its_parent_instead_of_spawning_again(self):
        with mock.patch.dict(os.environ, {flow._APP_CHILD_ENV: "7",
                                          "__CFBundleIdentifier": flow.APP_BUNDLE_ID}), \
                mock.patch.object(flow, "_macos_major", return_value=27), \
                mock.patch.object(flow.subprocess, "Popen") as popen, \
                mock.patch.object(flow.threading, "Thread") as thread:
            flow._run_app_in_child_if_needed()
            # ...and does not hand the marker on to processes it starts.
            self.assertNotIn(flow._APP_CHILD_ENV, os.environ)
        popen.assert_not_called()
        self.assertIs(thread.call_args.kwargs["target"], flow._quit_when_parent_exits)
        self.assertEqual(thread.call_args.kwargs["args"], (7,))
        self.assertTrue(thread.call_args.kwargs["daemon"])
        thread.return_value.start.assert_called_once_with()

    def test_the_child_quits_when_its_parent_goes_away(self):
        r, w = os.pipe()
        self.addCleanup(self._close_quietly, r)
        os.close(w)                               # what the parent dying does
        with mock.patch.object(flow.os, "kill") as kill, \
                redirect_stdout(io.StringIO()):
            flow._quit_when_parent_exits(r)
        kill.assert_called_once_with(os.getpid(), signal.SIGTERM)

    def test_compatibility_version_reads_as_the_real_release(self):
        for shown, major in (("27.0", 27), ("16.4", 26), ("15.6", 15), ("", 0)):
            with self.subTest(shown), \
                    mock.patch("platform.mac_ver",
                               return_value=(shown, ("", "", ""), "arm64")):
                self.assertEqual(flow._macos_major(), major)


if __name__ == "__main__":
    unittest.main()
