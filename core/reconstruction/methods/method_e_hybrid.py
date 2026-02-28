"""
Method E: Hybrid Photogrammetry.

Combines real photos with AI-generated views (SyncDreamer/Zero123) and
uses COLMAP for photogrammetry reconstruction.  Requires 6+ GB VRAM and
COLMAP to be installed.
"""

import time
from pathlib import Path
from typing import List, Optional

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod
from core.reconstruction.components.view_synthesizer import SyncDreamerSynthesizer
from core.reconstruction.components.colmap_wrapper import COLMAPWrapper
from core.reconstruction.components.mesh_verifier import MeshVerifier


class MethodEHybrid(BaseReconstructor):
    """Hybrid Photogrammetry reconstruction method.

    Workflow:
    1. Select the sharpest reference image from the real photos.
    2. Generate synthetic views using SyncDreamer / Zero123++.
    3. Combine real + synthetic images (20–25 total).
    4. Run COLMAP photogrammetry on the combined set.
    5. Verify and refine geometry using the real images.
    6. Return a high-quality mesh.

    The view synthesizer and COLMAP wrapper are constructed lazily on the
    first :meth:`reconstruct` call.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._view_synthesizer: Optional[SyncDreamerSynthesizer] = None
        self._colmap: Optional[COLMAPWrapper] = None
        self._verifier = MeshVerifier()
        self._num_synthetic_views = (
            config.get("method_e.num_synthetic_views", 16) if config else 16
        )

    def can_run(self) -> tuple:
        """Check whether Method E can run on the current hardware.

        Returns:
            ``(can_run: bool, reason: str)``
        """
        hw = MethodSelector.detect_hardware()
        if not hw.can_run_method_e:
            if not hw.has_gpu:
                return False, "Method E requires a GPU with 6+ GB VRAM (no GPU detected)"
            return (
                False,
                f"Method E requires 6+ GB VRAM (detected {hw.total_vram_gb:.1f} GB)",
            )
        if not MethodSelector.check_colmap_installed():
            return False, "Method E requires COLMAP to be installed"
        return True, "Ready"

    def estimate_time(self, num_images: int) -> int:
        """Estimate processing time for Method E.

        Args:
            num_images: Number of input images.

        Returns:
            Estimated seconds (scales with image count).
        """
        base = MethodSelector.get_method_requirements(ReconstructionMethod.METHOD_E)[
            "estimated_time_seconds"
        ]
        return base + (num_images - 3) * 60

    def reconstruct(self, images: List[Path], output_dir: Path) -> ReconstructionResult:
        """Run hybrid photogrammetry reconstruction.

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
            from PIL import Image as PILImage

            output_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: View synthesis
            self._update_progress(10, "Initializing view synthesizer…")
            if self._view_synthesizer is None:
                self._view_synthesizer = SyncDreamerSynthesizer(self.config)

            self._update_progress(15, "Selecting reference image…")
            reference_img = self._select_best_reference(images)

            self._update_progress(20, f"Generating {self._num_synthetic_views} synthetic views…")
            synthetic_frames = self._view_synthesizer.synthesize_multiple_views(
                reference_img,
                num_views=self._num_synthetic_views,
            )

            synth_dir = output_dir / "synthetic_views"
            synth_dir.mkdir(exist_ok=True)
            synth_paths: List[Path] = []
            for i, frame in enumerate(synthetic_frames):
                synth_path = synth_dir / f"synth_{i:03d}.jpg"
                PILImage.fromarray(frame).save(str(synth_path))
                synth_paths.append(synth_path)
                progress = 20 + int((i / max(1, len(synthetic_frames))) * 15)
                self._update_progress(
                    progress,
                    f"Generated view {i + 1}/{len(synthetic_frames)}",
                )

            # Stage 2: COLMAP photogrammetry
            self._update_progress(35, "Initializing COLMAP…")
            if self._colmap is None:
                self._colmap = COLMAPWrapper(self.config)

            all_images = list(images) + synth_paths
            self._update_progress(40, f"Running photogrammetry on {len(all_images)} images…")

            raw_mesh_path = self._colmap.run_full_pipeline(
                all_images,
                output_dir,
                progress_callback=lambda p, s: self._update_progress(
                    40 + int(p * 0.3), s
                ),
            )

            # Stage 3: Verification
            self._update_progress(70, "Verifying geometry with real images…")
            camera_poses = self._colmap.get_camera_poses()
            verified_path = self._verifier.verify_and_refine(
                raw_mesh_path,
                images,
                camera_poses,
                output_dir / "verified.obj",
            )

            # Stage 4: Quality score
            self._update_progress(85, "Evaluating quality…")
            quality_score = self._verifier.compute_quality_score(
                verified_path,
                images,
                camera_poses,
            )

            self._update_progress(100, "Complete!")

            return ReconstructionResult(
                success=True,
                mesh_path=verified_path,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start,
                quality_score=quality_score,
                metadata={
                    "num_real_images": len(images),
                    "num_synthetic_images": len(synth_paths),
                    "colmap_registered_images": len(camera_poses),
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
        return "Hybrid Photogrammetry (Method E)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Scoring weights / scales for reference image selection.
    _SHARPNESS_SCALE = 1000.0
    _SHARPNESS_WEIGHT = 0.6
    _CONTRAST_SCALE = 100.0
    _CONTRAST_WEIGHT = 0.4

    def _select_best_reference(self, images: List[Path]) -> Path:
        """Select the sharpest image as the view-synthesis reference.

        Scores each image by Laplacian variance (sharpness) and contrast.
        Falls back to the first image if OpenCV is not available.

        Args:
            images: Candidate image paths.

        Returns:
            Path to the best reference image.
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            return images[0]

        best_score = -1.0
        best_image = images[0]

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = float(gray.std())
            score = (
                (sharpness / self._SHARPNESS_SCALE) * self._SHARPNESS_WEIGHT
                + (contrast / self._CONTRAST_SCALE) * self._CONTRAST_WEIGHT
            )
            if score > best_score:
                best_score = score
                best_image = img_path

        return best_image
