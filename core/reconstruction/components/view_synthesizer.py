"""
ViewSynthesizer component.

Generates additional views of a subject from a sparse set of input images
using a view-synthesis model (Zero123++ or SyncDreamer).  Used by Method E.
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ViewSynthesizer:
    """Synthesize additional camera views from sparse input images.

    Delegates to :class:`SyncDreamerSynthesizer` for the actual view
    synthesis.  Supports Zero123++ (via ``diffusers``) or SyncDreamer as
    the underlying model, selected automatically at runtime.
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
            num_views: Total number of views to generate across all inputs.

        Returns:
            List of paths to synthesized view images.

        Raises:
            ImportError: If neither ``diffusers`` nor SyncDreamer is available.
        """
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError("Pillow is required for ViewSynthesizer — install with: pip install pillow") from exc

        synth = SyncDreamerSynthesizer(config=None)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[Path] = []

        views_per_image = max(1, num_views // max(1, len(images)))
        for img_path in images:
            frames = synth.synthesize_multiple_views(img_path, num_views=views_per_image)
            for i, frame in enumerate(frames):
                out = output_dir / f"{img_path.stem}_view_{i:03d}.jpg"
                PILImage.fromarray(frame).save(str(out))
                results.append(out)

        return results


class SyncDreamerSynthesizer:
    """View synthesizer backed by Zero123++ or SyncDreamer.

    The model is loaded lazily on the first :meth:`synthesize_multiple_views`
    call so that import time remains fast.
    """

    def __init__(self, config=None):
        self.config = config
        self._model = None
        self._device: Optional[str] = None
        self._backend: Optional[str] = None  # "zero123plus" or "syncdreamer"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize_multiple_views(
        self,
        reference_image: Path,
        num_views: int = 16,
    ) -> list:
        """Generate multiple novel views from a single reference image.

        Args:
            reference_image: Path to the reference input image.
            num_views: Number of novel views to generate.

        Returns:
            List of ``numpy.ndarray`` (H × W × 3) RGB images.

        Raises:
            ImportError: If no view-synthesis backend is available.
        """
        import numpy as np

        if self._model is None:
            self._load_model()

        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError("Pillow is required — install with: pip install pillow") from exc

        img = PILImage.open(str(reference_image)).convert("RGB").resize((256, 256))
        results = []

        import torch
        with torch.no_grad():
            for i in range(num_views):
                azimuth = 360.0 * i / num_views
                if self._backend == "zero123plus":
                    output = self._model(img, elevation=0.0, azimuth=azimuth)
                    frame = output.images[0] if hasattr(output, "images") else output
                else:
                    # SyncDreamer: generate all views at once and index into them
                    frames = self._model.generate(img, num_views=num_views)
                    results = [np.array(f) for f in frames]
                    return results
                results.append(np.array(frame))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the view-synthesis model.

        Tries Zero123++ (via ``diffusers``) first; falls back to SyncDreamer.

        Raises:
            ImportError: If neither backend is available.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for SyncDreamerSynthesizer — install with: pip install torch"
            ) from exc

        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        try:
            from diffusers import Zero123PlusPipeline  # type: ignore

            model_id = (
                self.config.get("view_synth.model_id", "sudo-ai/zero123plus-v1.1")
                if self.config else "sudo-ai/zero123plus-v1.1"
            )
            self._model = Zero123PlusPipeline.from_pretrained(model_id)
            self._model = self._model.to(self._device)
            self._backend = "zero123plus"
            logger.info("Loaded Zero123++ model on %s", self._device)
            return
        except ImportError:
            pass

        # Fallback: SyncDreamer (research code, expected in third_party/)
        try:
            import sys
            third_party = Path(__file__).parent.parent.parent.parent / "third_party" / "SyncDreamer"
            sys.path.insert(0, str(third_party))
            from ldm.models.diffusion.sync_dreamer import SyncDreamer  # type: ignore  # noqa: F401

            self._model = SyncDreamer()
            self._backend = "syncdreamer"
            logger.info("Loaded SyncDreamer model")
            return
        except ImportError:
            pass

        raise ImportError(
            "No view-synthesis model available. "
            "Install diffusers for Zero123++ support: pip install diffusers transformers accelerate"
        )
