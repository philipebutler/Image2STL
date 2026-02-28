"""
Method D: Dust3R Multi-View reconstruction stub.

Uses the Dust3R AI model for pairwise reconstruction followed by global
scene assembly and Poisson surface reconstruction.  Requires 4+ GB VRAM.
"""

import time
from pathlib import Path
from typing import List

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod


class MethodDDust3R(BaseReconstructor):
    """Dust3R multi-view reconstruction method (stub).

    Performs pairwise depth estimation across all image pairs, builds a
    global point cloud, then reconstructs the surface with Poisson meshing.
    """

    def can_run(self) -> tuple:
        """Check whether Method D can run on the current hardware.

        Returns:
            ``(can_run: bool, reason: str)``
        """
        hw = MethodSelector.detect_hardware()
        if not hw.can_run_method_d:
            if not hw.has_gpu:
                return False, "Method D requires a GPU with 4+ GB VRAM (no GPU detected)"
            return (
                False,
                f"Method D requires 4+ GB VRAM (detected {hw.total_vram_gb:.1f} GB)",
            )
        return True, "Ready"

    def estimate_time(self, num_images: int) -> int:
        """Estimate processing time for Method D.

        Args:
            num_images: Number of input images.

        Returns:
            Estimated seconds (scales with image pairs).
        """
        base = MethodSelector.get_method_requirements(ReconstructionMethod.METHOD_D)[
            "estimated_time_seconds"
        ]
        # Pairwise complexity: O(n^2)
        pairs = max(1, num_images * (num_images - 1) // 2)
        return base + pairs * 15

    def reconstruct(self, images: List[Path], output_dir: Path) -> ReconstructionResult:
        """Stub reconstruction — not yet implemented.

        Args:
            images: Input image paths.
            output_dir: Output directory.

        Returns:
            :class:`ReconstructionResult` indicating not-yet-implemented.
        """
        start = time.time()
        self._update_progress(0, "Method D: starting (stub)")
        return ReconstructionResult(
            success=False,
            mesh_path=None,
            method_used=self.get_method_name(),
            processing_time_seconds=time.time() - start,
            error_message="Not yet implemented",
        )

    def get_method_name(self) -> str:
        return "Dust3R Multi-View (Method D)"
