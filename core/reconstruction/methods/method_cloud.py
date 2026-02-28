"""
Method Cloud: Meshy.ai cloud reconstruction stub.

Adapts the existing Meshy.ai cloud API path to the ``BaseReconstructor``
interface.  Always available as the final fallback (no local GPU/CPU requirements).
Requires a valid Meshy.ai API key and an active internet connection.
"""

import time
from pathlib import Path
from typing import List

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod


class MethodCloud(BaseReconstructor):
    """Meshy.ai cloud reconstruction method (stub).

    Submits images to the Meshy.ai API for cloud-side reconstruction and
    downloads the resulting mesh.  Requires a valid API key stored in config.
    """

    def can_run(self) -> tuple:
        """Check whether the cloud method can run.

        An API key must be present in config for the cloud method to be
        considered runnable.

        Returns:
            ``(can_run: bool, reason: str)``
        """
        if self.config is None:
            return False, "No configuration provided"
        api_key = self.config.get("meshyApiKey", "") if hasattr(self.config, "get") else ""
        if not api_key:
            return False, "Meshy.ai API key not configured"
        return True, "Ready"

    def estimate_time(self, num_images: int) -> int:
        """Estimate processing time for cloud reconstruction.

        Args:
            num_images: Number of input images.

        Returns:
            Estimated seconds (dominated by API round-trip, not image count).
        """
        return MethodSelector.get_method_requirements(
            ReconstructionMethod.METHOD_CLOUD
        )["estimated_time_seconds"]

    def reconstruct(self, images: List[Path], output_dir: Path) -> ReconstructionResult:
        """Stub reconstruction — not yet implemented.

        Args:
            images: Input image paths.
            output_dir: Output directory.

        Returns:
            :class:`ReconstructionResult` indicating not-yet-implemented.
        """
        start = time.time()
        self._update_progress(0, "Method Cloud: starting (stub)")
        return ReconstructionResult(
            success=False,
            mesh_path=None,
            method_used=self.get_method_name(),
            processing_time_seconds=time.time() - start,
            error_message="Not yet implemented",
        )

    def get_method_name(self) -> str:
        return "Meshy.ai Cloud"
