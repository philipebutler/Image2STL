"""
Method C: TripoSR Fusion reconstruction stub.

Runs TripoSR on each input image independently to produce per-image meshes,
then aligns and fuses them into a single output mesh.  Runs on CPU so it is
always available as the last local fallback.
"""

import time
from pathlib import Path
from typing import List

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod


class MethodCTripoSR(BaseReconstructor):
    """TripoSR Fusion reconstruction method (stub).

    Generates one mesh per input image via TripoSR single-shot inference,
    aligns the results with ICP, and merges them into a final model.
    """

    def can_run(self) -> tuple:
        """Check whether Method C can run (always True — CPU-capable).

        Returns:
            ``(True, "Ready")``
        """
        # Method C is CPU-capable; no minimum VRAM requirement
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
        """Stub reconstruction — not yet implemented.

        Args:
            images: Input image paths.
            output_dir: Output directory.

        Returns:
            :class:`ReconstructionResult` indicating not-yet-implemented.
        """
        start = time.time()
        self._update_progress(0, "Method C: starting (stub)")
        return ReconstructionResult(
            success=False,
            mesh_path=None,
            method_used=self.get_method_name(),
            processing_time_seconds=time.time() - start,
            error_message="Not yet implemented",
        )

    def get_method_name(self) -> str:
        return "TripoSR Fusion (Method C)"
