"""
Tests for the reconstruction subsystem foundation (Phase 1).

Covers:
- HardwareCapabilities property logic
- MethodSelector.select_method() fallback chain ordering
- MethodSelector.get_method_requirements() data
- BaseReconstructor.validate_inputs() boundary checks
- ReconstructionResult dataclass initialisation
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from core.reconstruction.method_selector import (
    ReconstructionMethod,
    HardwareCapabilities,
    MethodSelector,
)
from core.reconstruction.base_reconstructor import (
    BaseReconstructor,
    ReconstructionResult,
)


# ---------------------------------------------------------------------------
# Concrete stub for testing the abstract BaseReconstructor
# ---------------------------------------------------------------------------

class _StubReconstructor(BaseReconstructor):
    """Minimal concrete subclass used only for testing."""

    def can_run(self):
        return True, "Ready"

    def estimate_time(self, num_images: int) -> int:
        return 60 * num_images

    def reconstruct(self, images, output_dir):
        return ReconstructionResult(
            success=True,
            mesh_path=output_dir / "stub.obj",
            method_used=self.get_method_name(),
            processing_time_seconds=0.0,
        )

    def get_method_name(self) -> str:
        return "Stub"


# ---------------------------------------------------------------------------
# HardwareCapabilities tests
# ---------------------------------------------------------------------------

class TestHardwareCapabilities(unittest.TestCase):
    """Tests for the HardwareCapabilities dataclass."""

    def test_has_gpu_with_cuda(self):
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=8.0)
        self.assertTrue(hw.has_gpu)

    def test_has_gpu_with_mps(self):
        hw = HardwareCapabilities(has_cuda=False, has_mps=True, total_vram_gb=8.0)
        self.assertTrue(hw.has_gpu)

    def test_no_gpu(self):
        hw = HardwareCapabilities(has_cuda=False, has_mps=False, total_vram_gb=0.0)
        self.assertFalse(hw.has_gpu)

    def test_can_run_method_e_requires_6gb(self):
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=6.0)
        self.assertTrue(hw.can_run_method_e)

    def test_cannot_run_method_e_insufficient_vram(self):
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=5.9)
        self.assertFalse(hw.can_run_method_e)

    def test_cannot_run_method_e_no_gpu(self):
        hw = HardwareCapabilities(has_cuda=False, has_mps=False, total_vram_gb=16.0)
        self.assertFalse(hw.can_run_method_e)

    def test_can_run_method_d_requires_4gb(self):
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=4.0)
        self.assertTrue(hw.can_run_method_d)

    def test_cannot_run_method_d_insufficient_vram(self):
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=3.9)
        self.assertFalse(hw.can_run_method_d)

    def test_can_run_method_c_always_true(self):
        hw = HardwareCapabilities(has_cuda=False, has_mps=False, total_vram_gb=0.0)
        self.assertTrue(hw.can_run_method_c)

    def test_default_field_values(self):
        hw = HardwareCapabilities(has_cuda=False, has_mps=False)
        self.assertEqual(hw.cuda_devices, [])
        self.assertEqual(hw.total_vram_gb, 0.0)
        self.assertEqual(hw.total_ram_gb, 0.0)
        self.assertEqual(hw.cpu_cores, 1)
        self.assertEqual(hw.platform, "")


# ---------------------------------------------------------------------------
# MethodSelector tests
# ---------------------------------------------------------------------------

class TestMethodSelector(unittest.TestCase):
    """Tests for MethodSelector fallback chain and requirements."""

    # -- select_method / fallback chain ----------------------------------------

    def test_full_gpu_fallback_chain(self):
        """System with 8 GB VRAM gets E → D → C → Cloud."""
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=8.0)
        chain = MethodSelector.select_method(hw)
        self.assertEqual(chain, [
            ReconstructionMethod.METHOD_E,
            ReconstructionMethod.METHOD_D,
            ReconstructionMethod.METHOD_C,
            ReconstructionMethod.METHOD_CLOUD,
        ])

    def test_mid_gpu_fallback_chain(self):
        """System with 5 GB VRAM skips E → gets D → C → Cloud."""
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=5.0)
        chain = MethodSelector.select_method(hw)
        self.assertEqual(chain, [
            ReconstructionMethod.METHOD_D,
            ReconstructionMethod.METHOD_C,
            ReconstructionMethod.METHOD_CLOUD,
        ])

    def test_cpu_only_fallback_chain(self):
        """CPU-only system gets C → Cloud."""
        hw = HardwareCapabilities(has_cuda=False, has_mps=False, total_vram_gb=0.0)
        chain = MethodSelector.select_method(hw)
        self.assertEqual(chain, [
            ReconstructionMethod.METHOD_C,
            ReconstructionMethod.METHOD_CLOUD,
        ])

    def test_user_preference_overrides_auto(self):
        """User preference is placed first, rest of chain follows."""
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=8.0)
        chain = MethodSelector.select_method(
            hw, user_preference=ReconstructionMethod.METHOD_C
        )
        self.assertEqual(chain[0], ReconstructionMethod.METHOD_C)
        # Remaining methods should still be present as fallbacks
        self.assertIn(ReconstructionMethod.METHOD_E, chain)
        self.assertIn(ReconstructionMethod.METHOD_CLOUD, chain)

    def test_user_preference_not_duplicated(self):
        """User preference method should not appear twice in chain."""
        hw = HardwareCapabilities(has_cuda=True, has_mps=False, total_vram_gb=8.0)
        chain = MethodSelector.select_method(
            hw, user_preference=ReconstructionMethod.METHOD_E
        )
        self.assertEqual(chain.count(ReconstructionMethod.METHOD_E), 1)

    def test_mps_gpu_detection(self):
        """Apple Silicon with 10 GB VRAM enables E and D."""
        hw = HardwareCapabilities(has_cuda=False, has_mps=True, total_vram_gb=10.0)
        chain = MethodSelector.select_method(hw)
        self.assertIn(ReconstructionMethod.METHOD_E, chain)
        self.assertIn(ReconstructionMethod.METHOD_D, chain)

    # -- get_method_requirements -----------------------------------------------

    def test_method_requirements_method_e(self):
        reqs = MethodSelector.get_method_requirements(ReconstructionMethod.METHOD_E)
        self.assertEqual(reqs["name"], "Hybrid Photogrammetry")
        self.assertEqual(reqs["min_vram_gb"], 6.0)
        self.assertTrue(reqs["requires_gpu"])
        self.assertTrue(reqs["requires_colmap"])

    def test_method_requirements_method_d(self):
        reqs = MethodSelector.get_method_requirements(ReconstructionMethod.METHOD_D)
        self.assertEqual(reqs["name"], "Dust3R Multi-View")
        self.assertEqual(reqs["min_vram_gb"], 4.0)
        self.assertTrue(reqs["requires_gpu"])
        self.assertFalse(reqs["requires_colmap"])

    def test_method_requirements_method_c(self):
        reqs = MethodSelector.get_method_requirements(ReconstructionMethod.METHOD_C)
        self.assertEqual(reqs["name"], "TripoSR Fusion")
        self.assertFalse(reqs["requires_gpu"])

    def test_method_requirements_cloud(self):
        reqs = MethodSelector.get_method_requirements(ReconstructionMethod.METHOD_CLOUD)
        self.assertEqual(reqs["name"], "Meshy.ai Cloud")
        self.assertFalse(reqs["requires_gpu"])

    def test_method_requirements_all_have_estimated_time(self):
        for method in ReconstructionMethod:
            reqs = MethodSelector.get_method_requirements(method)
            self.assertIn("estimated_time_seconds", reqs)
            self.assertIsInstance(reqs["estimated_time_seconds"], int)

    # -- check_colmap_installed ------------------------------------------------

    def test_check_colmap_not_installed(self):
        """COLMAP is unlikely to be installed in CI — expect False."""
        result = MethodSelector.check_colmap_installed()
        self.assertIsInstance(result, bool)

    # -- detect_hardware -------------------------------------------------------

    def test_detect_hardware_returns_capabilities(self):
        hw = MethodSelector.detect_hardware()
        self.assertIsInstance(hw, HardwareCapabilities)
        self.assertIsInstance(hw.has_cuda, bool)
        self.assertIsInstance(hw.has_mps, bool)
        self.assertIsInstance(hw.cpu_cores, int)
        self.assertGreaterEqual(hw.cpu_cores, 1)


# ---------------------------------------------------------------------------
# ReconstructionResult tests
# ---------------------------------------------------------------------------

class TestReconstructionResult(unittest.TestCase):
    """Tests for the ReconstructionResult dataclass."""

    def test_success_result(self):
        result = ReconstructionResult(
            success=True,
            mesh_path=Path("/tmp/mesh.obj"),
            method_used="TestMethod",
            processing_time_seconds=42.5,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.mesh_path, Path("/tmp/mesh.obj"))
        self.assertIsNone(result.error_message)
        self.assertIsNone(result.quality_score)
        self.assertEqual(result.metadata, {})

    def test_failure_result(self):
        result = ReconstructionResult(
            success=False,
            mesh_path=None,
            method_used="TestMethod",
            processing_time_seconds=1.0,
            error_message="GPU out of memory",
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.mesh_path)
        self.assertEqual(result.error_message, "GPU out of memory")

    def test_metadata_default(self):
        result = ReconstructionResult(
            success=True,
            mesh_path=Path("/tmp/out.obj"),
            method_used="M",
            processing_time_seconds=0.0,
        )
        self.assertIsInstance(result.metadata, dict)

    def test_metadata_custom(self):
        result = ReconstructionResult(
            success=True,
            mesh_path=Path("/tmp/out.obj"),
            method_used="M",
            processing_time_seconds=0.0,
            metadata={"faces": 1000},
        )
        self.assertEqual(result.metadata["faces"], 1000)


# ---------------------------------------------------------------------------
# BaseReconstructor tests (via _StubReconstructor)
# ---------------------------------------------------------------------------

class TestBaseReconstructor(unittest.TestCase):
    """Tests for validate_inputs and progress callback plumbing."""

    def setUp(self):
        self.reconstructor = _StubReconstructor(config=None)

    def test_validate_no_images(self):
        ok, msg = self.reconstructor.validate_inputs([])
        self.assertFalse(ok)
        self.assertIn("No images", msg)

    def test_validate_too_few_images(self):
        ok, msg = self.reconstructor.validate_inputs([Path("/a.jpg"), Path("/b.jpg")])
        self.assertFalse(ok)
        self.assertIn("3 images", msg)

    def test_validate_too_many_images(self):
        paths = [Path(f"/img{i}.jpg") for i in range(6)]
        ok, msg = self.reconstructor.validate_inputs(paths)
        self.assertFalse(ok)
        self.assertIn("5 images", msg)

    def test_validate_missing_file(self):
        paths = [Path("/nonexistent_abc.jpg")] * 3
        ok, msg = self.reconstructor.validate_inputs(paths)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_validate_valid_images(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i in range(3):
                p = Path(td) / f"img{i}.jpg"
                p.touch()
                paths.append(p)
            ok, msg = self.reconstructor.validate_inputs(paths)
            self.assertTrue(ok)
            self.assertEqual(msg, "")

    def test_progress_callback(self):
        received = []
        self.reconstructor.set_progress_callback(
            lambda p, s: received.append((p, s))
        )
        self.reconstructor._update_progress(50, "half way")
        self.assertEqual(received, [(50, "half way")])

    def test_progress_no_callback(self):
        # Should not raise when no callback is set
        self.reconstructor._update_progress(10, "silent")

    def test_can_run_stub(self):
        ok, reason = self.reconstructor.can_run()
        self.assertTrue(ok)

    def test_estimate_time_stub(self):
        self.assertEqual(self.reconstructor.estimate_time(3), 180)

    def test_get_method_name_stub(self):
        self.assertEqual(self.reconstructor.get_method_name(), "Stub")


# ---------------------------------------------------------------------------
# ReconstructionMethod enum tests
# ---------------------------------------------------------------------------

class TestReconstructionMethod(unittest.TestCase):
    """Tests for the ReconstructionMethod enum."""

    def test_enum_values(self):
        self.assertEqual(ReconstructionMethod.METHOD_E.value, "hybrid_photogrammetry")
        self.assertEqual(ReconstructionMethod.METHOD_D.value, "dust3r")
        self.assertEqual(ReconstructionMethod.METHOD_C.value, "triposr_fusion")
        self.assertEqual(ReconstructionMethod.METHOD_CLOUD.value, "meshy_cloud")

    def test_enum_from_value(self):
        self.assertEqual(
            ReconstructionMethod("hybrid_photogrammetry"),
            ReconstructionMethod.METHOD_E,
        )

    def test_all_methods_count(self):
        self.assertEqual(len(ReconstructionMethod), 4)


if __name__ == "__main__":
    unittest.main()
