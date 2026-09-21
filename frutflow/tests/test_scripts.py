"""Tests for the files AROUND flow.py: the watchdog's backoff logic, the
dependency lock, and the example config. Nothing here launches the app, touches
launchd, or installs anything — the watchdog is sourced (its loop is guarded)
and driven with stubbed process checks and a fake clock.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flow  # noqa: E402  imported after adding the repository root


class WatchdogBackoffTests(unittest.TestCase):
    HARNESS = r"""
source "$WATCHDOG"
LOG="$TMP/watchdog.log"; FLOW_LOG="$TMP/flow.log"
running=0; relaunches=0; slept=""
flow_is_running() { [ "$running" -eq 1 ]; }
relaunch_app() { relaunches=$((relaunches + 1)); }
pause() { slept="$slept $1"; }
"""

    def _run(self, script: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run(
                ["/bin/bash", "-c", self.HARNESS + script],
                env={"WATCHDOG": str(ROOT / "watchdog.sh"), "TMP": tmp,
                     "HOME": tmp, "PATH": "/usr/bin:/bin"},
                capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return dict(line.split("=", 1) for line in out.stdout.split() if "=" in line)

    def test_sourcing_the_watchdog_does_not_start_its_loop(self):
        # Reaching the echo at all proves `source` returned.
        self.assertEqual(self._run('echo "sourced=yes"'), {"sourced": "yes"})

    def test_a_single_sighting_does_not_reset_the_backoff(self):
        got = self._run("""
watchdog_tick 100; watchdog_tick 120; watchdog_tick 140
echo "after_three=$consecutive_relaunches"
running=1; watchdog_tick 160            # alive 20 s after the relaunch...
echo "one_sighting=$consecutive_relaunches"
running=0; watchdog_tick 180            # ...and crashed again
echo "crashed_again=$consecutive_relaunches relaunches=$relaunches"
""")
        self.assertEqual(got["after_three"], "3")
        self.assertEqual(got["one_sighting"], "3")      # used to drop to 0 here
        self.assertEqual(got["crashed_again"], "4")
        self.assertEqual(got["relaunches"], "4")

    def test_a_sustained_healthy_run_resets_it(self):
        got = self._run("""
watchdog_tick 100; watchdog_tick 120
running=1; watchdog_tick 140; watchdog_tick 200
echo "not_yet=$consecutive_relaunches"
watchdog_tick 260                       # 120 s of unbroken health
echo "reset=$consecutive_relaunches"
""")
        self.assertEqual(got["not_yet"], "2")
        self.assertEqual(got["reset"], "0")

    def test_a_crash_restarts_the_healthy_streak(self):
        got = self._run("""
watchdog_tick 100
running=1; watchdog_tick 120; watchdog_tick 200     # 80 s healthy
running=0; watchdog_tick 212                        # crash: streak void
running=1; watchdog_tick 224; watchdog_tick 300     # only 76 s since the crash
echo "count=$consecutive_relaunches"
""")
        self.assertEqual(got["count"], "2")

    def test_fifth_consecutive_relaunch_backs_off_for_five_minutes(self):
        got = self._run("""
for t in 100 120 140 160; do watchdog_tick $t; done
echo "before=$(echo $slept | tr ' ' ',')"
watchdog_tick 180
echo "after=$(echo $slept | tr ' ' ',')"
""")
        self.assertNotIn("300", got["before"].split(","))
        self.assertEqual(got["after"].split(",")[-1], "300")


class RequirementsLockTests(unittest.TestCase):
    @staticmethod
    def _norm(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    @staticmethod
    def _ver(text: str) -> tuple:
        return tuple(int(p) for p in re.findall(r"\d+", text)[:4])

    def _lock(self) -> dict:
        pins = {}
        for line in (ROOT / "requirements.lock").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, _, version = line.partition("==")
                self.assertTrue(version, f"not an exact pin: {line!r}")
                pins[self._norm(name)] = version
        return pins

    def test_lock_names_its_python(self):
        text = (ROOT / "requirements.lock").read_text()
        self.assertRegex(text, r"(?m)^# python: \d+\.\d+$")   # run.sh parses this

    def test_every_declared_requirement_is_pinned_within_its_bounds(self):
        pins = self._lock()
        for raw in (ROOT / "requirements.txt").read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            name = re.match(r"[A-Za-z0-9_.\-]+", line).group(0)
            key = self._norm(name)
            self.assertIn(key, pins, f"{name} is required but not in the lock")
            pinned = self._ver(pins[key])
            for op, bound in re.findall(r"(>=|<)\s*([0-9][0-9.]*)", line):
                if op == ">=":
                    self.assertGreaterEqual(pinned, self._ver(bound),
                                            f"{name} pin is below its floor")
                else:
                    self.assertLess(pinned, self._ver(bound),
                                    f"{name} pin is above its ceiling")


class ExampleConfigTests(unittest.TestCase):
    def test_example_config_matches_the_defaults(self):
        example = json.loads((ROOT / "config.example.json").read_text())
        self.assertEqual(example, flow.DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
