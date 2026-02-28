"""
Method E: Hybrid Photogrammetry stub.

Combines real photos with AI-generated views (SyncDreamer/Zero123) and
uses COLMAP for photogrammetry reconstruction.  Requires 6+ GB VRAM and
COLMAP to be installed.
"""

import time
from pathlib import Path
from typing import List

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod


class MethodEHybrid(BaseReconstructor):
    """Hybrid Photogrammetry reconstruction method (stub).

    Uses AI-generated view synthesis combined with real images for dense
    photogrammetry via COLMAP.  Highest quality but most demanding method.
    """

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
        """Stub reconstruction — not yet implemented.

        Args:
            images: Input image paths.
            output_dir: Output directory.

        Returns:
            :class:`ReconstructionResult` indicating not-yet-implemented.
        """
        start = time.time()
        self._update_progress(0, "Method E: starting (stub)")
        return ReconstructionResult(
            success=False,
            mesh_path=None,
            method_used=self.get_method_name(),
            processing_time_seconds=time.time() - start,
            error_message="Not yet implemented",
        )

    def get_method_name(self) -> str:
        return "Hybrid Photogrammetry (Method E)"
