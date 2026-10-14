"""
An interrupted task must resume from its checkpoint and give the same files.

A child process runs a task and is killed while MSV is in its OOS loop. The
task is then run again with resume=True and its outputs are compared, byte by
byte, with an uninterrupted run.

Run with:  python -m unittest tests.test_resume
"""
import filecmp
import logging
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("BLAS_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

import numpy as np                                   # noqa: E402

from src.data import have_snapshot                   # noqa: E402
from src.runner import run_task                   # noqa: E402

HAVE_DATA = have_snapshot()
ARGS = ("Commodities", "MONTHLY")
STRATEGIES = ["GMV", "MV", "MSV"]
OOS_ROWS = 30
FILES = ["oos_returns_gross.csv", "oos_returns_net.csv", "oos_turnover.csv", "best_params.csv",
         "dsr_trials.csv"] + [f"weights/{s}.csv" for s in STRATEGIES]

CHILD = """
import sys
sys.path.insert(0, r'{root}')
from src.runner import run_task
run_task('Commodities', 'MONTHLY', r'{out}', {strategies!r}, {rows}, False)
"""


class TestResume(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, "data snapshot not present")
    def test_killed_task_resumes_to_the_same_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean, cut = Path(tmp) / "clean", Path(tmp) / "cut"
            run_task(*ARGS, clean, STRATEGIES, OOS_ROWS, False)

            env = dict(os.environ, CHECKPOINT_S="1", BLAS_THREADS="1")
            code = CHILD.format(root=ROOT, out=cut, strategies=STRATEGIES, rows=OOS_ROWS)
            child = subprocess.Popen([sys.executable, "-c", code], env=env, cwd=ROOT,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ckpt = cut / "Commodities" / "MONTHLY" / "checkpoints" / "MSV.oos.npz"
            t_done, deadline = 0, time.time() + 300
            try:
                while time.time() < deadline and child.poll() is None:
                    if ckpt.exists():
                        try:
                            t_done = int(np.load(ckpt)["t_done"])
                        except Exception:
                            t_done = 0
                        if t_done >= 5:
                            break
                    time.sleep(0.5)
            finally:
                child.kill()                         # only the process started here
                child.wait()

            # the task was interrupted in the middle of the MSV OOS loop
            self.assertGreaterEqual(t_done, 5)
            self.assertLess(t_done, OOS_ROWS)
            self.assertFalse((cut / "Commodities" / "MONTHLY" / "oos_returns_gross.csv").exists())

            run_task(*ARGS, cut, STRATEGIES, OOS_ROWS, True)
            for name in FILES:
                with self.subTest(file=name):
                    self.assertTrue(filecmp.cmp(clean / "Commodities" / "MONTHLY" / name,
                                                cut / "Commodities" / "MONTHLY" / name, shallow=False))


class TestLockedFiles(unittest.TestCase):
    """A file that is locked for a moment must not stop the run."""

    def test_checkpoint_save_survives_a_transient_lock(self):
        from unittest import mock
        from src.runner import Checkpoint
        real, calls = os.replace, {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise PermissionError(13, "Permission denied")
            return real(src, dst)

        with tempfile.TemporaryDirectory() as tmp:
            ck = Checkpoint(Path(tmp), "X", every_s=0.0)
            with mock.patch("src.runner.os.replace", flaky), mock.patch("src.runner.time.sleep"):
                ck.save_oos(2, np.arange(4.0), np.ones((4, 3)), None)
            state = ck.load_oos()
            self.assertEqual(calls["n"], 4)
            self.assertEqual(state["t_done"], 2)
            np.testing.assert_array_equal(state["returns"], np.arange(4.0))

    def test_a_lock_that_persists_skips_the_save_without_raising(self):
        from unittest import mock
        from src.runner import Checkpoint

        def locked(src, dst):
            raise PermissionError(13, "Permission denied")

        with tempfile.TemporaryDirectory() as tmp:
            ck = Checkpoint(Path(tmp), "X", every_s=0.0)
            with mock.patch("src.runner.os.replace", locked), mock.patch("src.runner.time.sleep"):
                ck.save_oos(2, np.arange(4.0), np.ones((4, 3)), None)      # must not raise
            self.assertIsNone(ck.load_oos())


if __name__ == "__main__":
    unittest.main(verbosity=2)
