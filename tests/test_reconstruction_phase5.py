"""
Tests for Phase 5: Full Method Implementations.

Covers:
- Lazy model loading in MethodC/D/E (model is None at init)
- reconstruct() with mocked model backends
- COLMAPWrapper subprocess flow (mocked)
- MeshAligner ICP via trimesh (real)
- MeshVerifier quality scoring via trimesh (real)
- ReconstructionEngine _repair / _optimize / _scale_and_export
- MethodE _select_best_reference falls back gracefully without cv2
- ViewSynthesizer / SyncDreamerSynthesizer lazy loading
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

from core.reconstruction.base_reconstructor import ReconstructionResult
from core.reconstruction.components.colmap_wrapper import COLMAPWrapper
from core.reconstruction.components.mesh_aligner import MeshAligner
from core.reconstruction.components.mesh_verifier import MeshVerifier
from core.reconstruction.components.view_synthesizer import SyncDreamerSynthesizer, ViewSynthesizer
from core.reconstruction.engine import ReconstructionEngine
from core.reconstruction.methods.method_c_triposr import MethodCTripoSR
from core.reconstruction.methods.method_d_dust3r import MethodDDust3R
from core.reconstruction.methods.method_e_hybrid import MethodEHybrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box_mesh(path: Path) -> Path:
    """Write a simple watertight box mesh to *path* and return it."""
    mesh = trimesh.creation.box()
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path


def _make_real_images(tmp_dir: Path, count: int = 3) -> list:
    """Create *count* small PNG images and return their paths."""
    from PIL import Image as PILImage

    paths = []
    for i in range(count):
        p = tmp_dir / f"img_{i:02d}.png"
        img = PILImage.new("RGB", (64, 64), color=(i * 80, 100, 150))
        img.save(str(p))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Lazy loading — model is None at init
# ---------------------------------------------------------------------------

class TestLazyModelLoading(unittest.TestCase):
    """Verify that AI models are NOT loaded at construction time."""

    def test_method_c_model_none_at_init(self):
        m = MethodCTripoSR()
        self.assertIsNone(m._model)

    def test_method_d_model_none_at_init(self):
        m = MethodDDust3R()
        self.assertIsNone(m._model)

    def test_method_e_view_synthesizer_none_at_init(self):
        m = MethodEHybrid()
        self.assertIsNone(m._view_synthesizer)

    def test_method_e_colmap_none_at_init(self):
        m = MethodEHybrid()
        self.assertIsNone(m._colmap)

    def test_syncdreamer_model_none_at_init(self):
        s = SyncDreamerSynthesizer()
        self.assertIsNone(s._model)


# ---------------------------------------------------------------------------
# MethodCTripoSR reconstruct() with mocked model
# ---------------------------------------------------------------------------

class TestMethodCReconstructMocked(unittest.TestCase):
    """Test MethodCTripoSR.reconstruct() without a real TripoSR installation."""

    def _make_mock_model(self, tmp_dir: Path) -> MagicMock:
        """Build a mock that returns a tiny trimesh when .run() is called."""
        mock_model = MagicMock()
        box = trimesh.creation.box()
        mock_model.run.return_value = box
        return mock_model

    def test_reconstruct_succeeds_with_mocked_model(self):
        """Skip this test when torch is not installed (CI environment)."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available — skipping mocked model reconstruction test")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            images = _make_real_images(tmp_dir)
            output_dir = tmp_dir / "output"

            m = MethodCTripoSR()
            mock_model = self._make_mock_model(tmp_dir)
            m._model = mock_model
            m._device = None

            result = m.reconstruct(images, output_dir)

        self.assertTrue(result.success, msg=result.error_message)
        self.assertIsNotNone(result.mesh_path)
        self.assertEqual(result.method_used, m.get_method_name())

    def test_reconstruct_returns_failure_without_torch(self):
        """Without torch installed, reconstruct() returns a failure result."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            images = _make_real_images(tmp_dir)
            output_dir = tmp_dir / "output"

            m = MethodCTripoSR()
            # Leave _model=None so _initialize_model() is called and fails

            result = m.reconstruct(images, output_dir)

        # Without tsr or torch, initialization will fail and result is a failure
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

    def test_reconstruct_validates_inputs_before_model_load(self):
        """Passing an empty list must fail before any model interaction."""
        m = MethodCTripoSR()
        mock_init = MagicMock()
        m._initialize_model = mock_init

        result = m.reconstruct([], Path("/tmp/out"))

        mock_init.assert_not_called()
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)


# ---------------------------------------------------------------------------
# MethodDDust3R reconstruct() with mocked model
# ---------------------------------------------------------------------------

class TestMethodDReconstructMocked(unittest.TestCase):
    """Test MethodDDust3R.reconstruct() without a real Dust3R installation."""

    def test_reconstruct_validates_inputs_before_model_load(self):
        """Passing too few images must fail before any model interaction."""
        m = MethodDDust3R()
        mock_init = MagicMock()
        m._initialize_model = mock_init

        result = m.reconstruct([Path("/a.jpg"), Path("/b.jpg")], Path("/tmp/out"))

        mock_init.assert_not_called()
        self.assertFalse(result.success)

    def test_reconstruct_returns_failure_without_deps(self):
        """Without dust3r/torch, reconstruct() returns a failure result."""
        m = MethodDDust3R()

        with tempfile.TemporaryDirectory() as tmp:
            images = _make_real_images(Path(tmp))
            result = m.reconstruct(images, Path(tmp) / "out")

        # Without torch/dust3r, initialization or processing will fail
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

    def test_reconstruct_returns_failure_when_initialize_raises(self):
        """If _initialize_model raises ImportError, reconstruct() should wrap it."""
        m = MethodDDust3R()

        def _fail_init():
            raise ImportError("dust3r not installed")

        m._initialize_model = _fail_init

        with tempfile.TemporaryDirectory() as tmp:
            images = _make_real_images(Path(tmp))
            result = m.reconstruct(images, Path(tmp) / "out")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)


# ---------------------------------------------------------------------------
# MethodEHybrid helpers
# ---------------------------------------------------------------------------

class TestMethodEHelpers(unittest.TestCase):
    """Test MethodEHybrid helper methods."""

    def test_select_best_reference_returns_path_from_list(self):
        """_select_best_reference must return one of the supplied paths."""
        m = MethodEHybrid()
        with tempfile.TemporaryDirectory() as tmp:
            images = _make_real_images(Path(tmp))
            ref = m._select_best_reference(images)
        self.assertIn(ref, images)

    def test_select_best_reference_single_image(self):
        m = MethodEHybrid()
        with tempfile.TemporaryDirectory() as tmp:
            images = _make_real_images(Path(tmp), count=1)
            ref = m._select_best_reference(images)
        self.assertEqual(ref, images[0])

    def test_reconstruct_validates_inputs_before_model_load(self):
        m = MethodEHybrid()
        mock_init = MagicMock()
        m._view_synthesizer = mock_init

        result = m.reconstruct([], Path("/tmp/out"))

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)


# ---------------------------------------------------------------------------
# COLMAPWrapper subprocess flow
# ---------------------------------------------------------------------------

class TestCOLMAPWrapperSubprocess(unittest.TestCase):
    """Test COLMAPWrapper using mocked subprocess calls."""

    def test_run_sfm_calls_three_colmap_subcommands(self):
        """run_sfm() should invoke feature_extractor, exhaustive_matcher, mapper."""
        cw = COLMAPWrapper()
        called_cmds = []

        def _mock_run_colmap(args, timeout_seconds=300):
            called_cmds.append(args[0])

        cw._run_colmap = _mock_run_colmap

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            images = _make_real_images(tmp_dir)
            cw.run_sfm(images, tmp_dir / "ws")

        self.assertIn("feature_extractor", called_cmds)
        self.assertIn("exhaustive_matcher", called_cmds)
        self.assertIn("mapper", called_cmds)

    def test_run_dense_calls_three_colmap_subcommands(self):
        """run_dense() should invoke image_undistorter, patch_match_stereo, stereo_fusion."""
        cw = COLMAPWrapper()
        called_cmds = []

        def _mock_run_colmap(args, timeout_seconds=300):
            called_cmds.append(args[0])

        cw._run_colmap = _mock_run_colmap

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ws = tmp_dir / "ws"
            (ws / "images").mkdir(parents=True)
            (ws / "sparse" / "0").mkdir(parents=True)
            cw.run_dense(ws, tmp_dir / "out")

        self.assertIn("image_undistorter", called_cmds)
        self.assertIn("patch_match_stereo", called_cmds)
        self.assertIn("stereo_fusion", called_cmds)

    def test_run_colmap_raises_runtime_error_when_not_found(self):
        """_run_colmap() should raise RuntimeError if COLMAP binary is absent."""
        cw = COLMAPWrapper()
        with patch("subprocess.run", side_effect=FileNotFoundError("colmap: not found")):
            with self.assertRaises(RuntimeError) as ctx:
                cw._run_colmap(["feature_extractor"])
        self.assertIn("COLMAP", str(ctx.exception))

    def test_run_colmap_raises_on_nonzero_exit(self):
        cw = COLMAPWrapper()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error message"
        with patch("subprocess.run", return_value=mock_result):
            with self.assertRaises(RuntimeError):
                cw._run_colmap(["feature_extractor"])

    def test_get_camera_poses_returns_dict(self):
        cw = COLMAPWrapper()
        poses = cw.get_camera_poses()
        self.assertIsInstance(poses, dict)


# ---------------------------------------------------------------------------
# MeshAligner (real trimesh)
# ---------------------------------------------------------------------------

class TestMeshAlignerReal(unittest.TestCase):
    """Test MeshAligner with actual trimesh meshes."""

    def test_align_single_mesh_returns_same_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _make_box_mesh(Path(tmp) / "box.obj")
            out = Path(tmp) / "aligned.obj"
            ma = MeshAligner()
            result_path = ma.align([src], out)
            self.assertEqual(result_path, out)
            self.assertTrue(out.exists())

    def test_align_two_meshes_produces_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            box1 = _make_box_mesh(Path(tmp) / "box1.obj")
            box2 = _make_box_mesh(Path(tmp) / "box2.obj")
            out = Path(tmp) / "merged.obj"
            ma = MeshAligner()
            result_path = ma.align([box1, box2], out)
            self.assertEqual(result_path, out)
            self.assertTrue(out.exists())

    def test_align_empty_list_raises_value_error(self):
        ma = MeshAligner()
        with self.assertRaises(ValueError):
            ma.align([], Path("/tmp/out.obj"))

    def test_align_meshes_returns_same_count(self):
        ma = MeshAligner()
        meshes = [trimesh.creation.box(), trimesh.creation.box()]
        aligned = ma.align_meshes(meshes)
        self.assertEqual(len(aligned), 2)

    def test_align_meshes_empty_returns_empty(self):
        ma = MeshAligner()
        result = ma.align_meshes([])
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# MeshVerifier (real trimesh)
# ---------------------------------------------------------------------------

class TestMeshVerifierReal(unittest.TestCase):
    """Test MeshVerifier with actual trimesh meshes."""

    def test_score_watertight_mesh_is_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = _make_box_mesh(Path(tmp) / "box.stl")
            mv = MeshVerifier()
            score = mv.score(mesh_path, [])
            self.assertGreaterEqual(score, 0.4)

    def test_score_returns_float_in_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = _make_box_mesh(Path(tmp) / "box.stl")
            mv = MeshVerifier()
            score = mv.score(mesh_path, [])
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_is_watertight_box_is_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = _make_box_mesh(Path(tmp) / "box.stl")
            mv = MeshVerifier()
            result = mv.is_watertight(mesh_path)
            self.assertTrue(result)

    def test_score_raises_on_missing_file(self):
        mv = MeshVerifier()
        with self.assertRaises(Exception):
            mv.score(Path("/tmp/no_such_mesh_xyz.obj"), [])

    def test_verify_and_refine_produces_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            mesh_path = _make_box_mesh(tmp_dir / "box.obj")
            out_path = tmp_dir / "refined.obj"
            mv = MeshVerifier()
            result = mv.verify_and_refine(mesh_path, [], {}, out_path)
            self.assertTrue(result.exists())

    def test_compute_quality_score_delegates_to_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = _make_box_mesh(Path(tmp) / "box.stl")
            mv = MeshVerifier()
            s1 = mv.score(mesh_path, [])
            s2 = mv.compute_quality_score(mesh_path, [], {})
            self.assertAlmostEqual(s1, s2)


# ---------------------------------------------------------------------------
# ViewSynthesizer / SyncDreamerSynthesizer
# ---------------------------------------------------------------------------

class TestViewSynthesizer(unittest.TestCase):
    """Test ViewSynthesizer and SyncDreamerSynthesizer interfaces."""

    def test_syncdreamer_synthesizer_raises_without_deps(self):
        """_load_model() raises ImportError when torch is not available."""
        s = SyncDreamerSynthesizer()
        with patch.dict("sys.modules", {"torch": None, "diffusers": None}):
            with self.assertRaises((ImportError, Exception)):
                s._load_model()

    def test_view_synthesizer_synthesize_raises_without_model_deps(self):
        """synthesize() without torch/diffusers installed should raise."""
        vs = ViewSynthesizer()
        with self.assertRaises((ImportError, Exception)):
            vs.synthesize([Path("/tmp/img.jpg")], Path("/tmp/views"))

    def test_syncdreamer_model_none_before_load(self):
        s = SyncDreamerSynthesizer()
        self.assertIsNone(s._model)

    def test_syncdreamer_synthesize_with_mocked_model(self):
        """With a mocked model, synthesize_multiple_views should return frames."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")

        s = SyncDreamerSynthesizer()
        s._device = "cpu"
        s._backend = "zero123plus"

        # Create a mock model that returns PIL-like output
        from PIL import Image as PILImage

        mock_output = MagicMock()
        mock_output.images = [PILImage.new("RGB", (256, 256))]
        mock_model = MagicMock(return_value=mock_output)
        s._model = mock_model

        with tempfile.TemporaryDirectory() as tmp:
            images = _make_real_images(Path(tmp), count=1)
            with patch("torch.no_grad", MagicMock(return_value=MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            ))):
                frames = s.synthesize_multiple_views(images[0], num_views=2)

        self.assertEqual(len(frames), 2)
        for frame in frames:
            self.assertIsInstance(frame, np.ndarray)


# ---------------------------------------------------------------------------
# ReconstructionEngine post-processing — _repair / _optimize / _scale_and_export
# ---------------------------------------------------------------------------

class TestEnginePostProcessing(unittest.TestCase):
    """Test that _repair / _optimize / _scale_and_export use trimesh correctly."""

    def _engine(self):
        return ReconstructionEngine(config=None)

    def test_repair_produces_output_file(self):
        engine = self._engine()
        with tempfile.TemporaryDirectory() as tmp:
            src = _make_box_mesh(Path(tmp) / "src.obj")
            dst = Path(tmp) / "repaired.obj"
            result = engine._repair(src, dst)
            self.assertTrue(result.exists())

    def test_repair_returns_src_when_trimesh_unavailable(self):
        engine = self._engine()
        src = Path("/tmp/nonexistent.obj")
        dst = Path("/tmp/repaired_fake.obj")
        with patch.dict("sys.modules", {"trimesh": None, "trimesh.repair": None}):
            result = engine._repair(src, dst)
        self.assertEqual(result, src)

    def test_optimize_produces_output_file(self):
        engine = self._engine()
        with tempfile.TemporaryDirectory() as tmp:
            src = _make_box_mesh(Path(tmp) / "src.obj")
            dst = Path(tmp) / "optimized.obj"
            result = engine._optimize(src, dst)
        # Simplification may fall back if fast-simplification is absent; file should still exist or src returned
        self.assertIn(result, [dst, src])

    def test_optimize_returns_src_when_trimesh_unavailable(self):
        engine = self._engine()
        src = Path("/tmp/nonexistent.obj")
        dst = Path("/tmp/optimized_fake.obj")
        with patch.dict("sys.modules", {"trimesh": None}):
            result = engine._optimize(src, dst)
        self.assertEqual(result, src)

    def test_scale_and_export_produces_stl(self):
        engine = self._engine()
        with tempfile.TemporaryDirectory() as tmp:
            src = _make_box_mesh(Path(tmp) / "src.obj")
            dst = Path(tmp) / "final.stl"
            result = engine._scale_and_export(src, dst, 150.0)
            self.assertTrue(result.exists())

    def test_scale_and_export_scales_to_target_size(self):
        engine = self._engine()
        with tempfile.TemporaryDirectory() as tmp:
            src = _make_box_mesh(Path(tmp) / "src.obj")
            dst = Path(tmp) / "final.stl"
            engine._scale_and_export(src, dst, 150.0)
            mesh = trimesh.load(str(dst))
        extent = mesh.bounds[1] - mesh.bounds[0]
        max_dim = float(np.max(extent))
        self.assertAlmostEqual(max_dim, 150.0, delta=0.5)

    def test_scale_and_export_returns_src_when_trimesh_unavailable(self):
        engine = self._engine()
        src = Path("/tmp/nonexistent.obj")
        dst = Path("/tmp/final_fake.stl")
        with patch.dict("sys.modules", {"trimesh": None, "numpy": None}):
            result = engine._scale_and_export(src, dst, 150.0)
        self.assertEqual(result, src)

    def test_full_post_process_pipeline_order(self):
        """Verify repair → optimize → scale_and_export order is preserved."""
        engine = self._engine()
        order = []

        with patch.object(engine, "_repair", side_effect=lambda s, d: (order.append("repair") or s)), \
             patch.object(engine, "_optimize", side_effect=lambda s, d: (order.append("optimize") or s)), \
             patch.object(engine, "_scale_and_export", side_effect=lambda s, d, mm: (order.append("scale") or s)):
            engine._post_process(Path("/tmp/mesh.obj"), Path("/tmp/out"))

        self.assertEqual(order, ["repair", "optimize", "scale"])


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPhase5PackageImports(unittest.TestCase):
    def test_all_components_importable(self):
        from core.reconstruction.components.colmap_wrapper import COLMAPWrapper  # noqa: F401
        from core.reconstruction.components.mesh_aligner import MeshAligner  # noqa: F401
        from core.reconstruction.components.mesh_verifier import MeshVerifier  # noqa: F401
        from core.reconstruction.components.view_synthesizer import SyncDreamerSynthesizer, ViewSynthesizer  # noqa: F401

    def test_all_methods_importable(self):
        from core.reconstruction.methods.method_c_triposr import MethodCTripoSR  # noqa: F401
        from core.reconstruction.methods.method_d_dust3r import MethodDDust3R  # noqa: F401
        from core.reconstruction.methods.method_e_hybrid import MethodEHybrid  # noqa: F401


if __name__ == "__main__":
    unittest.main()
