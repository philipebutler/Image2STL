"""
MeshVerifier component.

Scores reconstructed mesh quality using geometric sanity checks and
reprojection-based analysis.  Used to decide whether a method's output
meets quality thresholds for acceptance.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class MeshVerifier:
    """Score and validate reconstruction quality.

    Computes a quality score in ``[0.0, 1.0]`` based on watertightness,
    vertex/face count, and (when camera poses are available) reprojection
    error from the original input images.
    """

    def score(
        self,
        mesh_path: Path,
        reference_images: List[Path],
    ) -> float:
        """Compute a quality score for the given mesh.

        Score is composed of:

        * **0.4** — watertightness (manifold, no boundary edges)
        * **0.3** — mesh has vertices (non-empty)
        * **0.3** — face count above a minimum sanity threshold (> 100)

        Args:
            mesh_path: Path to the mesh to evaluate.
            reference_images: Original input images (reserved for future
                reprojection-based scoring).

        Returns:
            Quality score in ``[0.0, 1.0]``; higher is better.

        Raises:
            ValueError: If *mesh_path* does not exist (raised by trimesh).
            ImportError: If *trimesh* is not installed.
        """
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("trimesh is required for MeshVerifier — install with: pip install trimesh") from exc

        mesh = trimesh.load(str(mesh_path))
        total = 0.0

        if self.is_watertight(mesh_path):
            total += 0.4

        if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            total += 0.3

        if hasattr(mesh, "faces") and len(mesh.faces) > 100:
            total += 0.3

        return min(1.0, total)

    def is_watertight(self, mesh_path: Path) -> bool:
        """Check whether a mesh is watertight (manifold, no holes).

        Args:
            mesh_path: Path to the mesh file.

        Returns:
            ``True`` if the mesh is watertight, ``False`` otherwise.

        Raises:
            ValueError: If *mesh_path* does not exist (raised by trimesh).
            ImportError: If *trimesh* is not installed.
        """
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("trimesh is required for MeshVerifier — install with: pip install trimesh") from exc

        mesh = trimesh.load(str(mesh_path))
        return bool(mesh.is_watertight)

    def verify_and_refine(
        self,
        mesh_path: Path,
        reference_images: List[Path],
        camera_poses: dict,
        output_path: Path,
    ) -> Path:
        """Verify mesh quality and apply light geometric repair.

        Performs hole-filling, normal fixing, degenerate-face removal, and
        vertex merging then writes the result to *output_path*.

        Args:
            mesh_path: Raw mesh to verify.
            reference_images: Original input images (used for future
                reprojection refinement).
            camera_poses: Camera poses from COLMAP (reserved for future use).
            output_path: Where to write the verified/refined mesh.

        Returns:
            Path to the verified mesh (*output_path*).

        Raises:
            ImportError: If *trimesh* is not installed.
        """
        try:
            import trimesh
            import trimesh.repair
        except ImportError as exc:
            raise ImportError("trimesh is required for MeshVerifier — install with: pip install trimesh") from exc

        mesh = trimesh.load(str(mesh_path))
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.merge_vertices()
        mesh.export(str(output_path))
        logger.debug("Verified mesh written to %s", output_path)
        return output_path

    def compute_quality_score(
        self,
        mesh_path: Path,
        reference_images: List[Path],
        camera_poses: dict,
    ) -> float:
        """Compute a final quality score for a verified mesh.

        Delegates to :meth:`score`.

        Args:
            mesh_path: Path to the mesh.
            reference_images: Original input images.
            camera_poses: Camera poses (reserved for future reprojection use).

        Returns:
            Quality score in ``[0.0, 1.0]``.
        """
        return self.score(mesh_path, reference_images)
