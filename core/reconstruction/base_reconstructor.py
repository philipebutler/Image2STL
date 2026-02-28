"""
Abstract base class for all reconstruction methods.

Every reconstruction method (E, D, C, Cloud) inherits from ``BaseReconstructor``
and implements :meth:`can_run`, :meth:`estimate_time`, :meth:`reconstruct`, and
:meth:`get_method_name`.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """Result from a reconstruction attempt."""

    success: bool
    mesh_path: Optional[Path]
    method_used: str
    processing_time_seconds: float
    error_message: Optional[str] = None
    quality_score: Optional[float] = None  # 0-1 if available
    metadata: dict = field(default_factory=dict)


class BaseReconstructor(ABC):
    """Abstract base class for reconstruction methods.

    All reconstruction methods must implement this interface.
    """

    def __init__(self, config):
        """Initialise reconstructor.

        Args:
            config: Application configuration object (supports ``config.get(key, default)``).
        """
        self.config = config
        self.progress_callback: Optional[Callable[[int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Set callback for progress updates.

        Args:
            callback: ``function(progress: int, status: str)``
        """
        self.progress_callback = callback

    def _update_progress(self, progress: int, status: str):
        """Emit a progress update if a callback is registered."""
        if self.progress_callback:
            self.progress_callback(progress, status)
        logger.info("%d%%: %s", progress, status)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def can_run(self) -> tuple:
        """Check whether this method can run on the current system.

        Returns:
            ``(can_run: bool, reason: str)``
        """

    @abstractmethod
    def estimate_time(self, num_images: int) -> int:
        """Estimate processing time in seconds.

        Args:
            num_images: Number of input images.

        Returns:
            Estimated seconds.
        """

    @abstractmethod
    def reconstruct(self, images: List[Path], output_dir: Path) -> ReconstructionResult:
        """Perform reconstruction.

        Args:
            images: List of input image paths.
            output_dir: Directory for output files.

        Returns:
            :class:`ReconstructionResult` with success/failure info.
        """

    @abstractmethod
    def get_method_name(self) -> str:
        """Return a human-readable method name."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def validate_inputs(self, images: List[Path]) -> tuple:
        """Validate input images.

        Args:
            images: List of image paths.

        Returns:
            ``(is_valid: bool, error_message: str)``
        """
        if not images:
            return False, "No images provided"

        if len(images) < 3:
            return False, "At least 3 images required"

        if len(images) > 5:
            return False, "Maximum 5 images supported"

        for img_path in images:
            if not img_path.exists():
                return False, f"Image not found: {img_path}"

        return True, ""
