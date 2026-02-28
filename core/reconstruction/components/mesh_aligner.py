"""
MeshAligner component.

Aligns multiple meshes produced from different source images or methods
using Iterative Closest Point (ICP) so they can be merged into a single
coherent model.  Used by Method C (TripoSR Fusion).
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class MeshAligner:
    """Align and merge multiple meshes into a single model.

    Uses trimesh ICP (``trimesh.registration.icp``) to register per-image
    TripoSR meshes to a common coordinate frame before fusion.
    """

    def align(
        self,
        meshes: List[Path],
        output_path: Path,
        max_iterations: int = 100,
    ) -> Path:
        """Align a list of mesh files and merge them into a single output.

        Args:
            meshes: Paths to input mesh files to align.  Must be non-empty.
            output_path: Destination for the merged aligned mesh.
            max_iterations: Maximum ICP iterations per alignment step.

        Returns:
            Path to the aligned and merged output mesh (``output_path``).

        Raises:
            ImportError: If *trimesh* is not installed.
            ValueError: If *meshes* is empty.
        """
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("trimesh is required for MeshAligner — install with: pip install trimesh") from exc

        if not meshes:
            raise ValueError("No meshes provided for alignment")

        loaded = [trimesh.load(str(p)) for p in meshes]

        if len(loaded) == 1:
            loaded[0].export(str(output_path))
            return output_path

        aligned = self.align_meshes(loaded, max_iterations=max_iterations)
        combined = trimesh.util.concatenate(aligned)
        combined.merge_vertices()
        combined.update_faces(combined.nondegenerate_faces())
        combined.export(str(output_path))
        return output_path

    def align_meshes(
        self,
        meshes: list,
        max_iterations: int = 100,
    ) -> list:
        """Align an in-memory list of ``trimesh.Trimesh`` objects via ICP.

        The first mesh is treated as the reference; subsequent meshes are
        transformed to align with it.

        Args:
            meshes: List of ``trimesh.Trimesh`` objects.
            max_iterations: Maximum ICP iterations per alignment step.

        Returns:
            List of aligned ``trimesh.Trimesh`` objects (same length as input).

        Raises:
            ImportError: If *trimesh* is not installed.
        """
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("trimesh is required for MeshAligner — install with: pip install trimesh") from exc

        if not meshes:
            return meshes

        reference = meshes[0]
        aligned = [reference]

        for mesh in meshes[1:]:
            n_pts = min(2000, len(mesh.vertices), len(reference.vertices))
            if n_pts < 3:
                aligned.append(mesh)
                continue
            pts_src = mesh.sample(n_pts)
            pts_ref = reference.sample(n_pts)
            matrix, _, _ = trimesh.registration.icp(
                pts_src,
                pts_ref,
                max_iterations=max_iterations,
            )
            mesh.apply_transform(matrix)
            aligned.append(mesh)

        return aligned
