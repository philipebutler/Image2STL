"""
Tests for the reconstruction method stubs and component interfaces (Phase 2).

Covers:
- Each method's can_run() under different hardware scenarios
- Each method's get_method_name() returns expected string
- Each method's estimate_time() returns a positive integer
- Each method's reconstruct() stub returns a not-implemented result
- Component stub interfaces are importable and raise NotImplementedError
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.reconstruction.method_selector import HardwareCapabilities, ReconstructionMethod
from core.reconstruction.base_reconstructor import ReconstructionResult

from core.reconstruction.methods.method_e_hybrid import MethodEHybrid
from core.reconstruction.methods.method_d_dust3r import MethodDDust3R
from core.reconstruction.methods.method_c_triposr import MethodCTripoSR
from core.reconstruction.methods.method_cloud import MethodCloud

from core.reconstruction.components.view_synthesizer import ViewSynthesizer
from core.reconstruction.components.colmap_wrapper import COLMAPWrapper
from core.reconstruction.components.mesh_aligner import MeshAligner
from core.reconstruction.components.mesh_verifier import MeshVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hw(has_cuda=False, has_mps=False, vram=0.0):
    return HardwareCapabilities(has_cuda=has_cuda, has_mps=has_mps, total_vram_gb=vram)


# ---------------------------------------------------------------------------
# MethodEHybrid tests
# ---------------------------------------------------------------------------

class TestMethodEHybrid(unittest.TestCase):
    """Tests for the Hybrid Photogrammetry method stub."""

    def _make(self, config=None):
        return MethodEHybrid(config=config)

    # --- get_method_name ---

    def test_get_method_name(self):
        m = self._make()
        self.assertIn("E", m.get_method_name())
        self.assertIsInstance(m.get_method_name(), str)

    # --- can_run ---

    def test_can_run_no_gpu(self):
        m = self._make()
        with patch("core.reconstruction.methods.method_e_hybrid.MethodSelector.detect_hardware",
                   return_value=_hw(has_cuda=False, has_mps=False, vram=0.0)):
            ok, reason = m.can_run()
        self.assertFalse(ok)
        self.assertIn("GPU", reason)

    def test_can_run_insufficient_vram(self):
        m = self._make()
        with patch("core.reconstruction.methods.method_e_hybrid.MethodSelector.detect_hardware",
                   return_value=_hw(has_cuda=True, vram=4.0)):
            ok, reason = m.can_run()
        self.assertFalse(ok)
        self.assertIn("VRAM", reason)

    def test_can_run_no_colmap(self):
        m = self._make()
        with patch("core.reconstruction.methods.method_e_hybrid.MethodSelector.detect_hardware",
                   return_value=_hw(has_cuda=True, vram=8.0)), \
             patch("core.reconstruction.methods.method_e_hybrid.MethodSelector.check_colmap_installed",
                   return_value=False):
            ok, reason = m.can_run()
        self.assertFalse(ok)
        self.assertIn("COLMAP", reason)

    def test_can_run_ready(self):
        m = self._make()
        with patch("core.reconstruction.methods.method_e_hybrid.MethodSelector.detect_hardware",
                   return_value=_hw(has_cuda=True, vram=8.0)), \
             patch("core.reconstruction.methods.method_e_hybrid.MethodSelector.check_colmap_installed",
                   return_value=True):
            ok, reason = m.can_run()
        self.assertTrue(ok)
        self.assertEqual(reason, "Ready")

    # --- estimate_time ---

    def test_estimate_time_positive(self):
        m = self._make()
        self.assertGreater(m.estimate_time(3), 0)

    def test_estimate_time_scales_with_images(self):
        m = self._make()
        self.assertGreater(m.estimate_time(5), m.estimate_time(3))

    # --- reconstruct (stub) ---

    def test_reconstruct_stub(self):
        m = self._make()
        result = m.reconstruct([], Path("/tmp"))
        self.assertIsInstance(result, ReconstructionResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.method_used, m.get_method_name())


# ---------------------------------------------------------------------------
# MethodDDust3R tests
# ---------------------------------------------------------------------------

class TestMethodDDust3R(unittest.TestCase):
    """Tests for the Dust3R multi-view method stub."""

    def _make(self, config=None):
        return MethodDDust3R(config=config)

    def test_get_method_name(self):
        m = self._make()
        self.assertIn("D", m.get_method_name())
        self.assertIsInstance(m.get_method_name(), str)

    def test_can_run_no_gpu(self):
        m = self._make()
        with patch("core.reconstruction.methods.method_d_dust3r.MethodSelector.detect_hardware",
                   return_value=_hw(has_cuda=False, vram=0.0)):
            ok, reason = m.can_run()
        self.assertFalse(ok)
        self.assertIn("GPU", reason)

    def test_can_run_insufficient_vram(self):
        m = self._make()
        with patch("core.reconstruction.methods.method_d_dust3r.MethodSelector.detect_hardware",
                   return_value=_hw(has_cuda=True, vram=3.0)):
            ok, reason = m.can_run()
        self.assertFalse(ok)
        self.assertIn("VRAM", reason)

    def test_can_run_ready(self):
        m = self._make()
        with patch("core.reconstruction.methods.method_d_dust3r.MethodSelector.detect_hardware",
                   return_value=_hw(has_cuda=True, vram=6.0)):
            ok, reason = m.can_run()
        self.assertTrue(ok)
        self.assertEqual(reason, "Ready")

    def test_estimate_time_positive(self):
        m = self._make()
        self.assertGreater(m.estimate_time(3), 0)

    def test_estimate_time_scales_with_images(self):
        m = self._make()
        self.assertGreater(m.estimate_time(5), m.estimate_time(3))

    def test_reconstruct_stub(self):
        m = self._make()
        result = m.reconstruct([], Path("/tmp"))
        self.assertIsInstance(result, ReconstructionResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.method_used, m.get_method_name())


# ---------------------------------------------------------------------------
# MethodCTripoSR tests
# ---------------------------------------------------------------------------

class TestMethodCTripoSR(unittest.TestCase):
    """Tests for the TripoSR Fusion method stub."""

    def _make(self, config=None):
        return MethodCTripoSR(config=config)

    def test_get_method_name(self):
        m = self._make()
        self.assertIn("C", m.get_method_name())
        self.assertIsInstance(m.get_method_name(), str)

    def test_can_run_always_true(self):
        """Method C is CPU-capable — must always report ready."""
        m = self._make()
        ok, reason = m.can_run()
        self.assertTrue(ok)
        self.assertEqual(reason, "Ready")

    def test_can_run_without_gpu(self):
        """Even with no GPU, Method C should be runnable."""
        m = self._make()
        ok, _ = m.can_run()
        self.assertTrue(ok)

    def test_estimate_time_positive(self):
        m = self._make()
        self.assertGreater(m.estimate_time(3), 0)

    def test_estimate_time_scales_with_images(self):
        m = self._make()
        self.assertGreater(m.estimate_time(5), m.estimate_time(3))

    def test_reconstruct_stub(self):
        m = self._make()
        result = m.reconstruct([], Path("/tmp"))
        self.assertIsInstance(result, ReconstructionResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.method_used, m.get_method_name())


# ---------------------------------------------------------------------------
# MethodCloud tests
# ---------------------------------------------------------------------------

class TestMethodCloud(unittest.TestCase):
    """Tests for the Meshy.ai Cloud method stub."""

    def _make(self, config=None):
        return MethodCloud(config=config)

    def test_get_method_name(self):
        m = self._make()
        self.assertIn("Cloud", m.get_method_name())
        self.assertIsInstance(m.get_method_name(), str)

    def test_can_run_no_config(self):
        m = self._make(config=None)
        ok, reason = m.can_run()
        self.assertFalse(ok)
        self.assertIn("configuration", reason.lower())

    def test_can_run_no_api_key(self):
        cfg = MagicMock()
        cfg.get.return_value = ""
        m = self._make(config=cfg)
        ok, reason = m.can_run()
        self.assertFalse(ok)
        self.assertIn("API key", reason)

    def test_can_run_with_api_key(self):
        cfg = MagicMock()
        cfg.get.return_value = "sk-test-key"
        m = self._make(config=cfg)
        ok, reason = m.can_run()
        self.assertTrue(ok)
        self.assertEqual(reason, "Ready")

    def test_estimate_time_positive(self):
        m = self._make()
        self.assertGreater(m.estimate_time(3), 0)

    def test_estimate_time_constant(self):
        """Cloud time should not vary with image count (API-dominated)."""
        m = self._make()
        self.assertEqual(m.estimate_time(3), m.estimate_time(5))

    def test_reconstruct_stub(self):
        m = self._make()
        result = m.reconstruct([], Path("/tmp"))
        self.assertIsInstance(result, ReconstructionResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.method_used, m.get_method_name())


# ---------------------------------------------------------------------------
# Component stub interface tests
# ---------------------------------------------------------------------------

class TestComponentImports(unittest.TestCase):
    """Verify all component stubs are importable and raise NotImplementedError."""

    def test_view_synthesizer_importable(self):
        vs = ViewSynthesizer()
        self.assertIsNotNone(vs)

    def test_view_synthesizer_raises(self):
        vs = ViewSynthesizer()
        with self.assertRaises(NotImplementedError):
            vs.synthesize([], Path("/tmp"))

    def test_colmap_wrapper_importable(self):
        cw = COLMAPWrapper()
        self.assertIsNotNone(cw)

    def test_colmap_wrapper_sfm_raises(self):
        cw = COLMAPWrapper()
        with self.assertRaises(NotImplementedError):
            cw.run_sfm([], Path("/tmp"))

    def test_colmap_wrapper_dense_raises(self):
        cw = COLMAPWrapper()
        with self.assertRaises(NotImplementedError):
            cw.run_dense(Path("/tmp"), Path("/tmp/out"))

    def test_mesh_aligner_importable(self):
        ma = MeshAligner()
        self.assertIsNotNone(ma)

    def test_mesh_aligner_raises(self):
        ma = MeshAligner()
        with self.assertRaises(NotImplementedError):
            ma.align([], Path("/tmp/out.obj"))

    def test_mesh_verifier_importable(self):
        mv = MeshVerifier()
        self.assertIsNotNone(mv)

    def test_mesh_verifier_score_raises(self):
        mv = MeshVerifier()
        with self.assertRaises(NotImplementedError):
            mv.score(Path("/tmp/mesh.obj"), [])

    def test_mesh_verifier_watertight_raises(self):
        mv = MeshVerifier()
        with self.assertRaises(NotImplementedError):
            mv.is_watertight(Path("/tmp/mesh.obj"))


# ---------------------------------------------------------------------------
# Package-level import test
# ---------------------------------------------------------------------------

class TestPackageImports(unittest.TestCase):
    """Verify that all Phase 2 classes are accessible via the package root."""

    def test_methods_importable_from_package(self):
        from core.reconstruction import (  # noqa: F401
            MethodEHybrid,
            MethodDDust3R,
            MethodCTripoSR,
            MethodCloud,
        )

    def test_components_importable_from_package(self):
        from core.reconstruction import (  # noqa: F401
            ViewSynthesizer,
            COLMAPWrapper,
            MeshAligner,
            MeshVerifier,
        )


if __name__ == "__main__":
    unittest.main()
