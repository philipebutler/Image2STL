"""
COLMAPWrapper component stub.

Manages COLMAP subprocess invocation for structure-from-motion (SfM) and
dense reconstruction.  Used by Method E.
"""

from pathlib import Path
from typing import List, Optional


class COLMAPWrapper:
    """Wrapper around the COLMAP CLI for photogrammetry reconstruction (stub).

    Full implementation will manage feature extraction, feature matching,
    sparse reconstruction, and optionally dense reconstruction via COLMAP
    subprocesses with timeout and progress monitoring.
    """

    def run_sfm(
        self,
        images: List[Path],
        workspace_dir: Path,
        timeout_seconds: int = 600,
    ) -> Path:
        """Run COLMAP Structure-from-Motion pipeline.

        Args:
            images: Input image paths.
            workspace_dir: COLMAP workspace directory.
            timeout_seconds: Maximum allowed runtime.

        Returns:
            Path to the sparse reconstruction output directory.

        Raises:
            NotImplementedError: Always — this is a stub.
        """
        raise NotImplementedError(
            "COLMAPWrapper.run_sfm() is not yet implemented. "
            "Full implementation will invoke the COLMAP CLI for SfM reconstruction."
        )

    def run_dense(
        self,
        workspace_dir: Path,
        output_dir: Path,
        timeout_seconds: int = 1200,
    ) -> Optional[Path]:
        """Run COLMAP dense reconstruction pipeline.

        Args:
            workspace_dir: COLMAP workspace from a prior SfM run.
            output_dir: Directory for dense point cloud output.
            timeout_seconds: Maximum allowed runtime.

        Returns:
            Path to dense point cloud, or ``None`` on failure.

        Raises:
            NotImplementedError: Always — this is a stub.
        """
        raise NotImplementedError(
            "COLMAPWrapper.run_dense() is not yet implemented. "
            "Full implementation will invoke COLMAP dense stereo and fusion."
        )
