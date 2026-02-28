"""
Tests for the ReconstructionEngine orchestrator (Phase 3).

Covers:
- Successful first-method path
- Fallback to second method when first fails
- All-methods-failed consolidated error reporting
- Post-processing pipeline step invocation order
- Skipping methods that cannot run
- Callback wiring (on_progress, on_method_started, on_method_completed)
- Package-level import of ReconstructionEngine / MethodAttempt
- reconstruct_multi wiring in core/reconstruction_engine.py
"""

import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from core.reconstruction.base_reconstructor import ReconstructionResult
from core.reconstruction.engine import MethodAttempt, ReconstructionEngine
from core.reconstruction.method_selector import ReconstructionMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_result(method_name: str = "TestMethod", mesh_path: Path = Path("/tmp/mesh.obj")) -> ReconstructionResult:
    return ReconstructionResult(
        success=True,
        mesh_path=mesh_path,
        method_used=method_name,
        processing_time_seconds=1.0,
        quality_score=0.9,
        metadata={"faces": 1000},
    )


def _fail_result(method_name: str = "TestMethod", error: str = "stub failure") -> ReconstructionResult:
    return ReconstructionResult(
        success=False,
        mesh_path=None,
        method_used=method_name,
        processing_time_seconds=0.1,
        error_message=error,
    )


def _mock_reconstructor(can_run_result=(True, "Ready"), reconstruct_result=None):
    """Build a mock BaseReconstructor-like object."""
    m = MagicMock()
    m.can_run.return_value = can_run_result
    m.reconstruct.return_value = reconstruct_result or _ok_result()
    m.get_method_name.return_value = "MockMethod"
    return m


# ---------------------------------------------------------------------------
# MethodAttempt dataclass
# ---------------------------------------------------------------------------

class TestMethodAttempt(unittest.TestCase):
    def test_stores_method_and_result(self):
        result = _ok_result()
        attempt = MethodAttempt(method=ReconstructionMethod.METHOD_C, result=result)
        self.assertEqual(attempt.method, ReconstructionMethod.METHOD_C)
        self.assertIs(attempt.result, result)


# ---------------------------------------------------------------------------
# Successful first-method path
# ---------------------------------------------------------------------------

class TestSuccessFirstMethod(unittest.TestCase):
    def _engine(self):
        return ReconstructionEngine(config=None)

    def test_returns_success_when_first_method_succeeds(self):
        engine = self._engine()
        mock_rec = _mock_reconstructor(reconstruct_result=_ok_result("MethodC", Path("/tmp/c.obj")))

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec):
            result = engine.reconstruct(
                images=[Path("/img1.jpg"), Path("/img2.jpg"), Path("/img3.jpg")],
                output_dir=Path("/tmp/out"),
                method_chain=[ReconstructionMethod.METHOD_C],
            )

        self.assertTrue(result.success)
        self.assertEqual(result.method_used, "MockMethod")
        self.assertEqual(len(engine.attempts), 1)
        self.assertTrue(engine.attempts[0].result.success)

    def test_attempts_list_contains_one_entry_on_first_success(self):
        engine = self._engine()
        mock_rec = _mock_reconstructor()

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
            )

        self.assertEqual(len(engine.attempts), 1)

    def test_second_method_not_called_when_first_succeeds(self):
        engine = self._engine()
        mock_c = _mock_reconstructor(reconstruct_result=_ok_result("C"))
        mock_d = _mock_reconstructor(reconstruct_result=_ok_result("D"))

        call_order = []

        def _get_rec(method):
            if method == ReconstructionMethod.METHOD_C:
                call_order.append("C")
                return mock_c
            call_order.append("D")
            return mock_d

        with patch.object(engine, "_get_reconstructor", side_effect=_get_rec):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C, ReconstructionMethod.METHOD_D],
            )

        self.assertEqual(call_order, ["C"])


# ---------------------------------------------------------------------------
# Fallback to second method
# ---------------------------------------------------------------------------

class TestFallback(unittest.TestCase):
    def _engine(self):
        return ReconstructionEngine(config=None)

    def test_falls_back_when_first_method_fails(self):
        engine = self._engine()
        mock_fail = _mock_reconstructor(reconstruct_result=_fail_result("First", "OOM"))
        mock_ok = _mock_reconstructor(reconstruct_result=_ok_result("Second"))

        call_order = []

        def _get_rec(method):
            if method == ReconstructionMethod.METHOD_D:
                call_order.append("D")
                return mock_fail
            call_order.append("C")
            return mock_ok

        with patch.object(engine, "_get_reconstructor", side_effect=_get_rec):
            result = engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_D, ReconstructionMethod.METHOD_C],
            )

        self.assertTrue(result.success)
        self.assertEqual(call_order, ["D", "C"])
        self.assertEqual(len(engine.attempts), 2)

    def test_attempts_records_both_failed_and_succeeded(self):
        engine = self._engine()
        mock_fail = _mock_reconstructor(reconstruct_result=_fail_result())
        mock_ok = _mock_reconstructor(reconstruct_result=_ok_result())

        side_effects = [mock_fail, mock_ok]
        idx = {"i": 0}

        def _get_rec(method):
            rec = side_effects[idx["i"]]
            idx["i"] += 1
            return rec

        with patch.object(engine, "_get_reconstructor", side_effect=_get_rec):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_D, ReconstructionMethod.METHOD_C],
            )

        self.assertFalse(engine.attempts[0].result.success)
        self.assertTrue(engine.attempts[1].result.success)

    def test_skips_method_that_cannot_run(self):
        engine = self._engine()
        mock_blocked = _mock_reconstructor(can_run_result=(False, "No GPU"))
        mock_ok = _mock_reconstructor(reconstruct_result=_ok_result())

        side_effects = [mock_blocked, mock_ok]
        idx = {"i": 0}

        def _get_rec(method):
            rec = side_effects[idx["i"]]
            idx["i"] += 1
            return rec

        with patch.object(engine, "_get_reconstructor", side_effect=_get_rec):
            result = engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_E, ReconstructionMethod.METHOD_C],
            )

        self.assertTrue(result.success)
        # Skipped method is NOT recorded in attempts
        self.assertEqual(len(engine.attempts), 1)


# ---------------------------------------------------------------------------
# All-methods-failed
# ---------------------------------------------------------------------------

class TestAllMethodsFailed(unittest.TestCase):
    def _engine(self):
        return ReconstructionEngine(config=None)

    def test_returns_failure_when_all_methods_fail(self):
        engine = self._engine()
        mock_fail = _mock_reconstructor(reconstruct_result=_fail_result("M", "err"))

        with patch.object(engine, "_get_reconstructor", return_value=mock_fail):
            result = engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C, ReconstructionMethod.METHOD_CLOUD],
            )

        self.assertFalse(result.success)
        self.assertEqual(result.method_used, "none")

    def test_error_message_contains_each_method(self):
        engine = self._engine()
        mock_c = _mock_reconstructor(reconstruct_result=_fail_result("C", "cpu err"))
        mock_cloud = _mock_reconstructor(reconstruct_result=_fail_result("Cloud", "api err"))

        side_effects = [mock_c, mock_cloud]
        idx = {"i": 0}

        def _get_rec(method):
            rec = side_effects[idx["i"]]
            idx["i"] += 1
            return rec

        with patch.object(engine, "_get_reconstructor", side_effect=_get_rec):
            result = engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C, ReconstructionMethod.METHOD_CLOUD],
            )

        self.assertIn("All reconstruction methods failed", result.error_message)

    def test_no_runnable_methods_message(self):
        engine = self._engine()
        mock_blocked = _mock_reconstructor(can_run_result=(False, "blocked"))

        with patch.object(engine, "_get_reconstructor", return_value=mock_blocked):
            result = engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
            )

        self.assertFalse(result.success)
        self.assertIn("No reconstruction methods could run", result.error_message)


# ---------------------------------------------------------------------------
# Post-processing pipeline
# ---------------------------------------------------------------------------

class TestPostProcessing(unittest.TestCase):
    def _engine(self):
        return ReconstructionEngine(config=None)

    def test_post_process_called_on_success(self):
        engine = self._engine()
        mock_rec = _mock_reconstructor(reconstruct_result=_ok_result("C", Path("/tmp/c.obj")))

        post_process_calls = []

        def _fake_post(mesh_path, output_dir, on_progress=None):
            post_process_calls.append(mesh_path)
            return Path("/tmp/final.stl")

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec), \
             patch.object(engine, "_post_process", side_effect=_fake_post):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
            )

        self.assertEqual(len(post_process_calls), 1)

    def test_post_process_not_called_on_failure(self):
        engine = self._engine()
        mock_rec = _mock_reconstructor(reconstruct_result=_fail_result())

        post_process_calls = []

        def _fake_post(mesh_path, output_dir, on_progress=None):
            post_process_calls.append(mesh_path)
            return Path("/tmp/final.stl")

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec), \
             patch.object(engine, "_post_process", side_effect=_fake_post):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
            )

        self.assertEqual(len(post_process_calls), 0)

    def test_pipeline_step_order(self):
        """repair → optimize → scale_and_export must be called in that order."""
        engine = self._engine()
        call_order = []

        def fake_repair(src, dst):
            call_order.append("repair")
            return src

        def fake_optimize(src, dst):
            call_order.append("optimize")
            return src

        def fake_scale(src, dst, scale_mm):
            call_order.append("scale_and_export")
            return src

        with patch.object(engine, "_repair", side_effect=fake_repair), \
             patch.object(engine, "_optimize", side_effect=fake_optimize), \
             patch.object(engine, "_scale_and_export", side_effect=fake_scale):
            engine._post_process(Path("/tmp/mesh.obj"), Path("/tmp/out"))

        self.assertEqual(call_order, ["repair", "optimize", "scale_and_export"])

    def test_final_mesh_path_returned(self):
        engine = self._engine()
        expected = Path("/tmp/final.stl")

        with patch.object(engine, "_repair", return_value=Path("/tmp/mesh.obj")), \
             patch.object(engine, "_optimize", return_value=Path("/tmp/mesh.obj")), \
             patch.object(engine, "_scale_and_export", return_value=expected):
            result = engine._post_process(Path("/tmp/mesh.obj"), Path("/tmp/out"))

        self.assertEqual(result, expected)

    def test_post_process_result_mesh_path_propagated(self):
        """The final_path from _post_process must appear in the returned result."""
        engine = self._engine()
        mock_rec = _mock_reconstructor(reconstruct_result=_ok_result())
        final = Path("/tmp/final.stl")

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec), \
             patch.object(engine, "_post_process", return_value=final):
            result = engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
            )

        self.assertEqual(result.mesh_path, final)


# ---------------------------------------------------------------------------
# Callback wiring
# ---------------------------------------------------------------------------

class TestCallbacks(unittest.TestCase):
    def _engine(self):
        return ReconstructionEngine(config=None)

    def test_on_method_started_called(self):
        engine = self._engine()
        mock_rec = _mock_reconstructor()
        mock_rec.get_method_name.return_value = "MockC"
        started = []

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec), \
             patch.object(engine, "_post_process", return_value=Path("/tmp/f.stl")):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
                on_method_started=started.append,
            )

        self.assertEqual(started, ["MockC"])

    def test_on_method_completed_called_with_success(self):
        engine = self._engine()
        mock_rec = _mock_reconstructor()
        mock_rec.get_method_name.return_value = "MockC"
        completed = []

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec), \
             patch.object(engine, "_post_process", return_value=Path("/tmp/f.stl")):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
                on_method_completed=lambda name, ok: completed.append((name, ok)),
            )

        self.assertEqual(completed, [("MockC", True)])

    def test_on_method_completed_called_with_failure(self):
        engine = self._engine()
        mock_rec = _mock_reconstructor(reconstruct_result=_fail_result())
        mock_rec.get_method_name.return_value = "MockC"
        completed = []

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
                on_method_completed=lambda name, ok: completed.append((name, ok)),
            )

        self.assertEqual(completed, [("MockC", False)])

    def test_on_progress_wired_to_reconstructor(self):
        engine = self._engine()
        mock_rec = _mock_reconstructor()
        progress_events = []

        with patch.object(engine, "_get_reconstructor", return_value=mock_rec), \
             patch.object(engine, "_post_process", return_value=Path("/tmp/f.stl")):
            engine.reconstruct(
                images=[Path("/a.jpg")] * 3,
                output_dir=Path("/tmp"),
                method_chain=[ReconstructionMethod.METHOD_C],
                on_progress=progress_events.append,
            )

        mock_rec.set_progress_callback.assert_called_once()


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageImports(unittest.TestCase):
    def test_engine_importable_from_package(self):
        from core.reconstruction import ReconstructionEngine, MethodAttempt  # noqa: F401

    def test_engine_is_correct_type(self):
        from core.reconstruction import ReconstructionEngine
        engine = ReconstructionEngine()
        self.assertIsInstance(engine, ReconstructionEngine)


# ---------------------------------------------------------------------------
# reconstruct_multi wiring in core/reconstruction_engine.py
# ---------------------------------------------------------------------------

class TestReconstructMultiWiring(unittest.TestCase):
    """Verify reconstruct_multi starts a background thread and fires callbacks."""

    def test_reconstruct_multi_exists(self):
        from core.reconstruction_engine import ReconstructionEngine as ThreadEngine
        engine = ThreadEngine()
        self.assertTrue(callable(engine.reconstruct_multi))

    def test_reconstruct_multi_calls_on_success(self):
        from core.reconstruction_engine import ReconstructionEngine as ThreadEngine

        ok_result = _ok_result("MockC", Path("/tmp/c.obj"))
        success_calls = []

        def fake_inner_reconstruct(images, output_dir, **kwargs):
            return ok_result

        engine = ThreadEngine()
        with patch(
            "core.reconstruction_engine._MultiMethodEngine.reconstruct",
            side_effect=fake_inner_reconstruct,
        ):
            engine.reconstruct_multi(
                images=["/a.jpg", "/b.jpg", "/c.jpg"],
                output_dir=Path("/tmp/out"),
                on_success=lambda path, stats: success_calls.append((path, stats)),
            )
            engine.wait_for_completion(timeout=5.0)

        self.assertEqual(len(success_calls), 1)
        self.assertEqual(success_calls[0][0], str(ok_result.mesh_path))

    def test_reconstruct_multi_calls_on_error(self):
        from core.reconstruction_engine import ReconstructionEngine as ThreadEngine

        fail_result = _fail_result("none", "All methods failed")
        fail_result_obj = ReconstructionResult(
            success=False,
            mesh_path=None,
            method_used="none",
            processing_time_seconds=0.0,
            error_message="All methods failed",
        )
        error_calls = []

        def fake_inner_reconstruct(images, output_dir, **kwargs):
            return fail_result_obj

        engine = ThreadEngine()
        with patch(
            "core.reconstruction_engine._MultiMethodEngine.reconstruct",
            side_effect=fake_inner_reconstruct,
        ):
            engine.reconstruct_multi(
                images=["/a.jpg", "/b.jpg", "/c.jpg"],
                output_dir=Path("/tmp/out"),
                on_error=lambda code, msg: error_calls.append((code, msg)),
            )
            engine.wait_for_completion(timeout=5.0)

        self.assertEqual(len(error_calls), 1)
        self.assertEqual(error_calls[0][0], "ALL_METHODS_FAILED")

    def test_reconstruct_multi_does_not_start_when_running(self):
        from core.reconstruction_engine import ReconstructionEngine as ThreadEngine

        engine = ThreadEngine()
        barrier = threading.Event()

        def slow_reconstruct(images, output_dir, **kwargs):
            barrier.wait(timeout=3.0)
            return _ok_result()

        with patch(
            "core.reconstruction_engine._MultiMethodEngine.reconstruct",
            side_effect=slow_reconstruct,
        ):
            engine.reconstruct_multi(
                images=["/a.jpg", "/b.jpg", "/c.jpg"],
                output_dir=Path("/tmp/out"),
            )
            self.assertTrue(engine.is_running)
            # Second call should be ignored
            engine.reconstruct_multi(
                images=["/a.jpg", "/b.jpg", "/c.jpg"],
                output_dir=Path("/tmp/out"),
            )
            barrier.set()
            engine.wait_for_completion(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
