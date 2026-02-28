"""
MeshAligner component stub.

Aligns multiple meshes produced from different source images or methods
using Iterative Closest Point (ICP) so they can be merged into a single
coherent model.  Used by Method C (TripoSR Fusion).
"""

from pathlib import Path
from typing import List


class MeshAligner:
    """Align and merge multiple meshes into a single model (stub).

    Full implementation will use trimesh or Open3D ICP alignment to register
    per-image TripoSR meshes to a common coordinate frame before fusion.
    """

    def align(
        self,
        meshes: List[Path],
        output_path: Path,
        max_iterations: int = 100,
    ) -> Path:
        """Align a list of meshes and merge them into a single output.

        Args:
            meshes: Paths to input mesh files to align.
            output_path: Destination for the merged aligned mesh.
            max_iterations: Maximum ICP iterations per alignment step.

        Returns:
            Path to the aligned and merged output mesh.

        Raises:
            NotImplementedError: Always — this is a stub.
        """
        raise NotImplementedError(
            "MeshAligner.align() is not yet implemented. "
            "Full implementation will use ICP via trimesh/Open3D for mesh alignment."
        )
