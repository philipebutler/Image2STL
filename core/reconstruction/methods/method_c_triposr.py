"""
Method C: TripoSR Fusion reconstruction.

Runs TripoSR on each input image independently to produce per-image meshes,
then aligns and fuses them into a single output mesh.  Runs on CPU so it is
always available as the last local fallback.
"""

import time
from pathlib import Path
from typing import List, Optional

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod
from core.reconstruction.components.mesh_aligner import MeshAligner


class MethodCTripoSR(BaseReconstructor):
    """TripoSR Fusion reconstruction method.

    Generates one mesh per input image via TripoSR single-shot inference,
    aligns the results with ICP, and merges them into a final model.

    The TripoSR model is loaded lazily on the first :meth:`reconstruct` call
    so that startup time remains fast.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._device: Optional[str] = None
        self._aligner = MeshAligner()

    def can_run(self) -> tuple:
        """Check whether Method C can run (always True — CPU-capable).

        Returns:
            ``(True, "Ready")``
        """
        return True, "Ready"

    def estimate_time(self, num_images: int) -> int:
        """Estimate processing time for Method C.

        Args:
            num_images: Number of input images.

        Returns:
            Estimated seconds (linear with image count).
        """
        base = MethodSelector.get_method_requirements(ReconstructionMethod.METHOD_C)[
            "estimated_time_seconds"
        ]
        return base + (num_images - 3) * 45

    def reconstruct(self, images: List[Path], output_dir: Path) -> ReconstructionResult:
        """Run TripoSR fusion reconstruction.

        Pipeline:
        1. Validate inputs.
        2. Load TripoSR model (lazy).
        3. Run TripoSR on each image to produce individual meshes.
        4. Align meshes via ICP.
        5. Fuse aligned meshes and clean up.

        Args:
            images: Input image paths.
            output_dir: Output directory.

        Returns:
            :class:`ReconstructionResult` with success/failure details.
        """
        start = time.time()

        is_valid, error = self.validate_inputs(images)
        if not is_valid:
            return ReconstructionResult(
                success=False,
                mesh_path=None,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start,
                error_message=error,
            )

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Load model
            self._update_progress(5, "Loading TripoSR model…")
            if self._model is None:
                self._initialize_model()

            # Stage 2: Per-image reconstruction
            import torch

            individual_dir = output_dir / "individual_meshes"
            individual_dir.mkdir(exist_ok=True)
            meshes = []

            for i, img_path in enumerate(images):
                progress_start = 10 + i * 60 // len(images)
                progress_end = 10 + (i + 1) * 60 // len(images)
                self._update_progress(
                    progress_start,
                    f"Reconstructing from image {i + 1}/{len(images)}…",
                )

                from PIL import Image as PILImage
                img = PILImage.open(str(img_path)).convert("RGB")
                img_tensor = self._preprocess_image(img)

                with torch.no_grad():
                    mesh = self._model.run(img_tensor)

                mesh_path = individual_dir / f"mesh_{i:02d}.obj"
                mesh.export(str(mesh_path))
                meshes.append(mesh)

                self._update_progress(progress_end, f"Completed mesh {i + 1}/{len(images)}")

            # Stage 3: Align
            self._update_progress(70, "Aligning meshes…")
            aligned_meshes = self._aligner.align_meshes(meshes)

            # Stage 4: Fuse
            self._update_progress(85, "Fusing meshes…")
            fused = self._fuse_meshes(aligned_meshes)

            # Stage 5: Cleanup
            self._update_progress(95, "Cleaning up mesh…")
            fused.merge_vertices()
            fused.fill_holes()
            fused.update_faces(fused.nondegenerate_faces())

            final_path = output_dir / "triposr_fused.obj"
            fused.export(str(final_path))
            self._update_progress(100, "Complete!")

            return ReconstructionResult(
                success=True,
                mesh_path=final_path,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start,
                metadata={
                    "num_meshes_fused": len(meshes),
                    "final_vertices": len(fused.vertices),
                    "final_faces": len(fused.faces),
                },
            )

        except Exception as exc:
            return ReconstructionResult(
                success=False,
                mesh_path=None,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start,
                error_message=str(exc),
            )

    def get_method_name(self) -> str:
        return "TripoSR Fusion (Method C)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_model(self) -> None:
        """Lazy-load the TripoSR model.

        Raises:
            ImportError: If the ``tsr`` package is not installed.
        """
        try:
            import torch
            from tsr.system import TSR  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TripoSR is not installed. "
                "Install with: pip install tsr  (or from https://github.com/VAST-AI-Research/TripoSR)"
            ) from exc

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        model_path = (
            self.config.get("method_c.model_path", "stabilityai/TripoSR")
            if self.config else "stabilityai/TripoSR"
        )
        self._model = TSR.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

    def _preprocess_image(self, img):
        """Preprocess a PIL image into a tensor for TripoSR."""
        import numpy as np
        import torch

        img = img.resize((512, 512))
        arr = np.array(img).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        if self._device is not None:
            tensor = tensor.to(self._device)
        return tensor

    @staticmethod
    def _fuse_meshes(meshes: list):
        """Concatenate aligned trimesh objects into one combined mesh."""
        import numpy as np
        import trimesh

        all_vertices, all_faces = [], []
        offset = 0
        for mesh in meshes:
            all_vertices.append(mesh.vertices)
            all_faces.append(mesh.faces + offset)
            offset += len(mesh.vertices)

        return trimesh.Trimesh(
            vertices=np.vstack(all_vertices),
            faces=np.vstack(all_faces),
            process=False,
        )
