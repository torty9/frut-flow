"""Failure-path regressions. No microphone, clipboard, or key events are used."""

import io
import os
import queue
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flow


class RecorderFailureTests(unittest.TestCase):
    def recorder(self):
        recorder = flow.Recorder.__new__(flow.Recorder)
        recorder._lock = threading.Lock()
        recorder._frames_lock = threading.Lock()
        recorder._frames = [flow.np.array([[0.1], [0.2]], dtype=flow.np.float32)]
        recorder._stream = mock.Mock()
        recorder.recording = True
        return recorder

    def test_device_disconnect_during_stop_keeps_captured_audio(self):
        for failing_method in ("stop", "close"):
            with self.subTest(failing_method=failing_method):
                recorder = self.recorder()
                stream = recorder._stream
                getattr(stream, failing_method).side_effect = RuntimeError("device disconnected")
                with redirect_stdout(io.StringIO()):
                    audio = recorder.stop()
                flow.np.testing.assert_allclose(audio, [0.1, 0.2])
                stream.close.assert_called_once()
                self.assertFalse(recorder.recording)
                self.assertIsNone(recorder._stream)
                self.assertEqual(recorder._frames, [])

    def test_failed_stream_start_closes_handle_before_retry(self):
        recorder = self.recorder()
        recorder.recording = False
        recorder._stream = None
        recorder.sample_rate = flow.SAMPLE_RATE
        first, second = mock.Mock(), mock.Mock()
        first.start.side_effect = RuntimeError("device asleep")
        events = []
        first.close.side_effect = lambda: events.append("close failed stream")
        recorder._sd = mock.Mock()
        recorder._sd.InputStream.side_effect = [first, second]
        recorder._sd._terminate.side_effect = lambda: events.append("terminate")
        recorder.start()
        self.assertEqual(events, ["close failed stream", "terminate"])
        self.assertIs(recorder._stream, second)
        self.assertTrue(recorder.recording)


class CaptureOrderingTests(unittest.TestCase):
    def test_stop_cannot_overtake_press_at_state_lock_release(self):
        unlocked = threading.Event()
        resume_press = threading.Event()

        class YieldAfterUnlock:
            def __init__(self):
                self.lock = threading.RLock()

            def __enter__(self):
                self.lock.acquire()

            def __exit__(self, *_args):
                self.lock.release()
                if threading.current_thread().name == "test-press":
                    unlocked.set()
                    resume_press.wait(2)

        app = flow.FlowApp.__new__(flow.FlowApp)
        app._state_lock = YieldAfterUnlock()
        app._capture_q = queue.SimpleQueue()
        app._capture_requested = False
        app._capture_generation = 0
        app._locked = False
        press = threading.Thread(target=lambda: app._request_capture(True),
                                 name="test-press", daemon=True)
        press.start()
        try:
            self.assertTrue(unlocked.wait(2))
            app._request_capture(False)
        finally:
            resume_press.set()
            press.join(2)
        self.assertFalse(press.is_alive())
        self.assertEqual(app._capture_q.get_nowait(), (1, True))
        self.assertEqual(app._capture_q.get_nowait(), (2, False))


class UndoSafetyTests(unittest.TestCase):
    def setUp(self):
        self.app = flow.FlowApp.__new__(flow.FlowApp)
        self.app.cfg = dict(flow.DEFAULT_CONFIG, play_sounds=False,
                            learn_from_edits=False, learn_vocab=False)
        self.app._state_lock = threading.RLock()
        self.app._pending_learn = None
        self.app._learn_timer = None
        self.app._undo_stack = [(" Dictation.", "Editor", "field")]
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(redirect_stdout(io.StringIO()))
        for name, value in (("_focused_app_name", "Editor"),
                            ("_ax_focused_element", "field"),
                            ("_ax_read_value", "Earlier text. Dictation."),
                            ("_ax_trusted", True), ("_secure_input_active", False)):
            self.patches.enter_context(mock.patch.object(flow, name, return_value=value))
        self.selection = self.patches.enter_context(mock.patch.object(
            flow, "_ax_read_selected_range", create=True, return_value=(24, 0)))
        self.backspaces = self.patches.enter_context(mock.patch.object(flow, "_send_backspaces"))
        self.history = self.patches.enter_context(mock.patch.object(flow, "pop_history_matching"))
        self.patches.enter_context(mock.patch.object(flow, "play"))

    def test_cursor_moved_inside_same_field_refuses_undo(self):
        self.selection.return_value = (5, 0)
        self.assertFalse(self.app._undo_last_insertion())
        self.backspaces.assert_not_called()
        self.assertEqual(len(self.app._undo_stack), 1)

    def test_selected_text_refuses_undo(self):
        self.selection.return_value = (0, 24)
        self.assertFalse(self.app._undo_last_insertion())
        self.backspaces.assert_not_called()

    def test_unreadable_selection_refuses_undo(self):
        self.selection.return_value = None
        self.assertFalse(self.app._undo_last_insertion())
        self.backspaces.assert_not_called()

    def test_unknown_original_field_refuses_undo(self):
        self.app._undo_stack = [(" Dictation.", "Editor", None)]
        self.assertFalse(self.app._undo_last_insertion())
        self.backspaces.assert_not_called()

    def test_unchanged_dictation_at_cursor_can_be_undone(self):
        self.assertTrue(self.app._undo_last_insertion())
        self.backspaces.assert_called_once_with(11)
        self.assertEqual(self.app._undo_stack, [])

    def test_cursor_offsets_use_utf16_for_emoji(self):
        value = "😀 Dictation."
        with mock.patch.object(flow, "_ax_read_value", return_value=value):
            self.selection.return_value = (len(value.encode("utf-16-le")) // 2, 0)
            self.assertTrue(self.app._undo_last_insertion())
        self.backspaces.assert_called_once_with(11)

    def test_undo_continuation_does_not_log_private_text(self):
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(flow, "insert_text", return_value=False):
            self.app._record_history = mock.Mock()
            self.app._record_usage_stats = mock.Mock()
            self.app._apply_undo_result("Private sentence", 0)
        self.assertNotIn("Private sentence", output.getvalue())

    def test_clipboard_only_continuation_is_recoverable_in_history(self):
        self.app._record_history = mock.Mock()
        self.app._record_usage_stats = mock.Mock()
        with mock.patch.object(flow, "insert_text", return_value=False):
            self.app._apply_undo_result("Keep this", 0)
        self.app._record_history.assert_called_once_with(
            "Keep this", app="Editor", delivered=False)

    def test_backspaces_clear_held_modifiers(self):
        quartz = mock.Mock()
        quartz.CGEventCreateKeyboardEvent.side_effect = lambda *_args: {"flags": 0x100000}
        quartz.CGEventSetFlags.side_effect = lambda event, flags: event.update(flags=flags)
        posted = []
        quartz.CGEventPost.side_effect = lambda _tap, event: posted.append(event.copy())
        # Use the original function; setUp patches the external delivery boundary.
        with mock.patch.dict(sys.modules, {"Quartz": quartz}), mock.patch.object(flow.time, "sleep"):
            self.send_backspaces(2)
        self.assertEqual(len(posted), 4)
        self.assertTrue(all(event["flags"] == 0 for event in posted))

    send_backspaces = staticmethod(flow._send_backspaces)


class ClipboardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(flow.DEFAULT_CONFIG)
        self.count = 0
        self.text = "Original clipboard"
        self.timers = []
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(mock.patch.object(flow, "_CLIPBOARD_RESTORE", None, create=True))
        self.patches.enter_context(mock.patch.object(flow, "_ax_trusted", return_value=True))
        self.patches.enter_context(mock.patch.object(flow, "_secure_input_active", return_value=False))
        self.paste = self.patches.enter_context(mock.patch.object(flow, "_send_paste"))
        self.patches.enter_context(mock.patch.object(flow, "_clip_snapshot", side_effect=lambda: ("text", self.text)))
        self.patches.enter_context(mock.patch.object(flow, "_clip_set", side_effect=self.write))
        self.patches.enter_context(mock.patch.object(flow, "_clip_change_count", side_effect=lambda: self.count))
        self.restore = self.patches.enter_context(mock.patch.object(
            flow, "_clip_restore", side_effect=lambda snapshot: self.write(snapshot[1])))
        self.patches.enter_context(mock.patch.object(flow.threading, "Timer", side_effect=self.timer))

    def write(self, text):
        self.count += 1
        self.text = text
        return self.count

    def timer(self, _delay, callback):
        self.timers.append(callback)
        return mock.Mock()

    def test_rapid_pastes_restore_original_clipboard(self):
        flow.insert_text("First dictation", self.cfg)
        flow.insert_text("Second dictation", self.cfg)
        for callback in self.timers:
            callback()
        self.assertEqual(self.text, "Original clipboard")

    def test_new_user_copy_wins_over_pending_restore(self):
        flow.insert_text("Dictation", self.cfg)
        self.write("New user copy")
        for callback in self.timers:
            callback()
        self.assertEqual(self.text, "New user copy")

    def test_user_copy_between_dictations_becomes_restoration_target(self):
        flow.insert_text("First", self.cfg)
        self.write("New user copy")
        flow.insert_text("Second", self.cfg)
        for callback in self.timers:
            callback()
        self.assertEqual(self.text, "New user copy")

    def test_unknown_clipboard_generation_never_overwrites_new_copy(self):
        with mock.patch.object(flow, "_clip_set", return_value=-1):
            flow.insert_text("Dictation", self.cfg)
        self.write("New user copy")
        for callback in self.timers:
            callback()
        self.assertEqual(self.text, "New user copy")

    def test_paste_event_failure_preserves_clipboard_fallback(self):
        self.paste.side_effect = RuntimeError("could not post key event")
        with redirect_stdout(io.StringIO()):
            self.assertFalse(flow.insert_text("Dictation", self.cfg))
        self.assertEqual(self.text, "Dictation")
        self.assertEqual(self.timers, [])


class UndoPhraseTests(unittest.TestCase):
    def test_mentions_of_commands_are_preserved(self):
        for text in ("Do not delete that.", "I said never mind.",
                     "Please don't forget that.", "No digas borra eso."):
            with self.subTest(text=text):
                self.assertEqual(flow.apply_undo(text, flow.DEFAULT_CONFIG), (text, 0))

    def test_explicit_inline_clause_retracts_one_sentence(self):
        self.assertEqual(flow.apply_undo(
            "Keep this. Remove this. Actually never mind.", flow.DEFAULT_CONFIG),
            ("Keep this.", 0))

    def test_leading_command_keeps_continuation(self):
        self.assertEqual(flow.apply_undo(
            "Never mind. Use this instead.", flow.DEFAULT_CONFIG),
            ("Use this instead.", 1))


class ClipboardIOTests(unittest.TestCase):
    def test_empty_clipboard_can_be_restored(self):
        board = mock.Mock()
        board.pasteboardItems.return_value = []
        with mock.patch.object(flow, "_pasteboard", return_value=board):
            snapshot = flow._clip_snapshot()
            self.assertEqual(snapshot, ("items", []))
            flow._clip_restore(snapshot)
        board.clearContents.assert_called_once()
        board.writeObjects_.assert_not_called()

    def test_rejected_native_write_uses_checked_fallback(self):
        board = mock.Mock()
        board.setString_forType_.return_value = False
        with mock.patch.object(flow, "_pasteboard", return_value=board), \
                mock.patch.object(flow, "_pbcopy") as copy, \
                mock.patch.object(flow, "_CLIPBOARD_RESTORE", (5, ("text", "old"))):
            self.assertEqual(flow._clip_set("Dictation"), -1)
            self.assertIsNone(flow._CLIPBOARD_RESTORE)
        copy.assert_called_once_with("Dictation")

    def test_failing_copy_command_is_not_reported_as_success(self):
        with mock.patch.object(flow, "PBCOPY", "/usr/bin/false"):
            with self.assertRaises(subprocess.CalledProcessError):
                flow._pbcopy("Dictation")

    def test_typing_failure_keeps_transcript_on_clipboard(self):
        keyboard = types.ModuleType("pynput.keyboard")
        keyboard.Controller = mock.Mock()
        keyboard.Controller.return_value.type.side_effect = RuntimeError("cannot type")
        with mock.patch.dict(sys.modules, {"pynput.keyboard": keyboard}), \
                mock.patch.object(flow, "_ax_trusted", return_value=True), \
                mock.patch.object(flow, "_secure_input_active", return_value=False), \
                mock.patch.object(flow, "_clip_set") as copy, \
                redirect_stdout(io.StringIO()):
            self.assertFalse(flow.insert_text(
                "Dictation", dict(flow.DEFAULT_CONFIG, insert_method="type")))
        copy.assert_called_once_with("Dictation")


class FileTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(flow.DEFAULT_CONFIG, learn_vocab=False, fuzzy_correct=False)

    def test_missing_file_writes_error_only_to_stderr(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(stdout), redirect_stderr(stderr):
            code = flow.transcribe_file(self.cfg, str(Path(tmp) / "missing.wav"))
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("file not found", stderr.getvalue())

    def test_backend_and_cleanup_diagnostics_do_not_pollute_transcript(self):
        def build(_cfg):
            print("Backend warning")
            return mock.Mock(transcribe=mock.Mock(return_value="raw"))

        def clean(*_args):
            print("Cleanup warning")
            return "Actual transcript."

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(flow, "_load_audio_file", return_value=flow.np.zeros(1600)), \
                mock.patch.object(flow, "build_transcriber", side_effect=build), \
                mock.patch.object(flow, "clean", side_effect=clean), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = flow.transcribe_file(self.cfg, "memo.wav")
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "Actual transcript.\n")
        self.assertIn("Backend warning", stderr.getvalue())
        self.assertIn("Cleanup warning", stderr.getvalue())

    def test_inference_failure_returns_error_status_without_traceback(self):
        model = mock.Mock()
        model.transcribe.side_effect = RuntimeError("inference failed")
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(flow, "_load_audio_file", return_value=flow.np.zeros(1600)), \
                mock.patch.object(flow, "build_transcriber", return_value=model), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = flow.transcribe_file(self.cfg, "memo.wav")
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("inference failed", stderr.getvalue())


class InstallerPythonTests(unittest.TestCase):
    def test_bundle_uses_venv_interpreter_even_if_path_python_is_newer(self):
        installer = (ROOT / "Install frut Flow.command").read_text()
        function = installer.split("create_app_bundle() {", 1)[1].split(
            "\ncreate_launch_agent() {", 1)[0]
        function = "create_app_bundle() {" + function
        for existing_version in (None, "3.12", "3.11"):
            with self.subTest(existing_version=existing_version), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                install = root / "installed app"
                venv_bin = install / ".venv/bin"
                venv_bin.mkdir(parents=True)
                shim_bin = root / "bin"
                shim_bin.mkdir()
                base = root / "venv-base-python"
                newer = root / "newer-system-python"
                base.write_text('#!/bin/bash\necho 3.12\n')
                newer.write_text('#!/bin/bash\necho 3.14\n')
                (venv_bin / "python").write_text(
                    '#!/bin/bash\ncase "$2" in\n'
                    '  *sys.version_info*) echo 3.12 ;;\n'
                    '  *) printf "%s\\n" "$TEST_VENV_BASE" ;;\nesac\n')
                (shim_bin / "python3").write_text(
                    '#!/bin/bash\ncase "$2" in\n'
                    '  *sys.version_info*) echo 3.14 ;;\n'
                    '  *) printf "%s\\n" "$TEST_NEWER_PYTHON" ;;\nesac\n')
                (shim_bin / "codesign").write_text('#!/bin/bash\nexit 0\n')
                for executable in (base, newer, venv_bin / "python",
                                   shim_bin / "python3", shim_bin / "codesign"):
                    executable.chmod(0o755)
                app = root / "frutflow.app"
                embedded = app / "Contents/MacOS/python3"
                if existing_version:
                    embedded.parent.mkdir(parents=True)
                    embedded.write_text(f"#!/bin/bash\necho {existing_version}\n")
                    embedded.chmod(0o755)
                env = dict(os.environ, APP=str(app), INSTALL_DIR=str(install),
                           TEST_VENV_BASE=str(base), TEST_NEWER_PYTHON=str(newer),
                           PATH=str(shim_bin) + ":/usr/bin:/bin")
                subprocess.run(["/bin/bash", "-c", function + "\ncreate_app_bundle"],
                               env=env, check=True, capture_output=True, text=True)
                self.assertEqual(embedded.read_bytes(), base.read_bytes())


if __name__ == "__main__":
    unittest.main()
