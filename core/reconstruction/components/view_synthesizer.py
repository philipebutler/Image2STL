"""
ViewSynthesizer component stub.

Generates additional views of a subject from a sparse set of input images
using a view-synthesis model (SyncDreamer / Zero123++).  Used by Method E.
"""

from pathlib import Path
from typing import List


class ViewSynthesizer:
    """Synthesize additional camera views from sparse input images (stub).

    Full implementation will integrate SyncDreamer or Zero123++ to generate
    novel views that augment the real photographs before COLMAP processing.
    """

    def synthesize(
        self,
        images: List[Path],
        output_dir: Path,
        num_views: int = 8,
    ) -> List[Path]:
        """Generate novel views for the given images.

        Args:
            images: Source input images.
            output_dir: Directory to write synthesized views.
            num_views: Number of views to generate per input image.

        Returns:
            List of paths to synthesized view images.

        Raises:
            NotImplementedError: Always — this is a stub.
        """
        raise NotImplementedError(
            "ViewSynthesizer.synthesize() is not yet implemented. "
            "Full implementation will use SyncDreamer/Zero123++ for view synthesis."
        )
