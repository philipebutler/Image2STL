"""
MeshVerifier component stub.

Scores reconstructed mesh quality using reprojection-based analysis and
geometric sanity checks.  Used to decide whether a method's output meets
quality thresholds for acceptance.
"""

from pathlib import Path
from typing import List


class MeshVerifier:
    """Score and validate reconstruction quality (stub).

    Full implementation will reproject mesh vertices back into input image
    space and measure reprojection error as a proxy for geometric accuracy,
    along with surface completeness and watertightness checks.
    """

    def score(
        self,
        mesh_path: Path,
        reference_images: List[Path],
    ) -> float:
        """Compute a quality score for the given mesh.

        Args:
            mesh_path: Path to the mesh to evaluate.
            reference_images: Original input images for reprojection.

        Returns:
            Quality score in [0.0, 1.0]; higher is better.

        Raises:
            NotImplementedError: Always — this is a stub.
        """
        raise NotImplementedError(
            "MeshVerifier.score() is not yet implemented. "
            "Full implementation will compute reprojection-based quality metrics."
        )

    def is_watertight(self, mesh_path: Path) -> bool:
        """Check whether a mesh is watertight (manifold, no holes).

        Args:
            mesh_path: Path to the mesh file.

        Returns:
            ``True`` if the mesh is watertight.

        Raises:
            NotImplementedError: Always — this is a stub.
        """
        raise NotImplementedError(
            "MeshVerifier.is_watertight() is not yet implemented. "
            "Full implementation will use trimesh/Open3D for watertightness checks."
        )
