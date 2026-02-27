"""Foreground isolation preprocessing using rembg."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


def _compute_settings_hash(
    strength: float,
    hole_fill: bool,
    island_threshold: int,
    crop_padding: int,
) -> str:
    """Compute a short hash of preprocessing parameters for deterministic naming."""
    key = f"{strength:.3f}|{hole_fill}|{island_threshold}|{crop_padding}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def _processed_name(source_path: Path, settings_hash: str) -> str:
    """Generate a deterministic output filename for a processed image."""
    return f"{source_path.stem}_{settings_hash}_processed.png"


def preprocess_image(
    source_path: Path,
    output_dir: Path,
    strength: float = 0.5,
    hole_fill: bool = True,
    island_removal_threshold: int = 500,
    crop_padding: int = 10,
) -> Path:
    """Isolate the foreground in an image and save the result as RGBA PNG.

    Args:
        source_path: Path to source image.
        output_dir: Directory to write processed images.
        strength: Background removal strength (controls post-processing aggressiveness).
        hole_fill: Whether to fill holes in the mask.
        island_removal_threshold: Minimum connected component size to keep.
        crop_padding: Padding (in pixels) to add around the tight crop.

    Returns:
        Path to the processed RGBA PNG file.

    Raises:
        ImportError: If rembg is not installed.
    """
    try:
        from rembg import remove as rembg_remove  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "rembg is not installed. Install it with: pip install rembg"
        ) from exc

    try:
        from PIL import Image
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("Pillow is required for preprocessing") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    settings_hash = _compute_settings_hash(
        strength, hole_fill, island_removal_threshold, crop_padding
    )
    output_name = _processed_name(source_path, settings_hash)
    output_path = output_dir / output_name

    # Cache hit: return immediately if already processed with same settings
    if output_path.exists():
        return output_path

    with Image.open(source_path) as img:
        if strength > 0.0:
            # Use alpha matting with conservative defaults to avoid noisy/grainy
            # foreground edges. Strength still controls aggressiveness, but we
            # keep thresholds in a high-confidence range.
            strength_clamped = max(0.0, min(1.0, float(strength)))
            fg_threshold = int(210 + (strength_clamped * 35))
            bg_threshold = int(8 + ((1.0 - strength_clamped) * 10))
            erode_size = int(max(1, math.ceil((1.0 - strength_clamped) * 4)))

            try:
                rgba = rembg_remove(
                    img,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=fg_threshold,
                    alpha_matting_background_threshold=bg_threshold,
                    alpha_matting_erode_size=erode_size,
                )
            except TypeError:
                # Older rembg versions may not support all alpha-matting kwargs.
                rgba = rembg_remove(
                    img,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=fg_threshold,
                )
        else:
            rgba = rembg_remove(img)

    rgba = _postprocess_mask(rgba, hole_fill, island_removal_threshold, crop_padding)
    rgba.save(str(output_path), format="PNG")
    return output_path


def _postprocess_mask(rgba_image, hole_fill: bool, island_threshold: int, crop_padding: int):
    """Apply mask cleanup, tight crop, and square pad to an RGBA image."""
    try:
        import numpy as np
        from PIL import Image
    except (ImportError, ModuleNotFoundError):
        return rgba_image

    arr = np.array(rgba_image)
    original_alpha = arr[:, :, 3].copy()
    cleanup_mask = (original_alpha > 10).astype(np.uint8) * 255

    if hole_fill:
        cleanup_mask = _fill_holes(cleanup_mask)
    if island_threshold > 0:
        cleanup_mask = _remove_small_islands(cleanup_mask, island_threshold)

    original_alpha = _smooth_alpha_channel(original_alpha)
    keep_region = cleanup_mask > 0
    arr[:, :, 3] = np.where(keep_region, original_alpha, 0).astype(original_alpha.dtype)
    arr = _tight_crop_and_square_pad(arr, crop_padding)
    return Image.fromarray(arr, mode="RGBA")


def _smooth_alpha_channel(alpha_channel):
    """Apply light denoising to alpha while preserving edge softness."""
    try:
        from scipy import ndimage  # type: ignore

        smoothed = ndimage.gaussian_filter(alpha_channel.astype(float), sigma=0.7)
        return smoothed.clip(0, 255).astype(alpha_channel.dtype)
    except (ImportError, ModuleNotFoundError):
        try:
            import numpy as np
            from PIL import Image, ImageFilter

            pil_alpha = Image.fromarray(alpha_channel, mode="L")
            pil_alpha = pil_alpha.filter(ImageFilter.GaussianBlur(radius=0.8))
            return np.array(pil_alpha).astype(alpha_channel.dtype)
        except (ImportError, ModuleNotFoundError):
            return alpha_channel


def _fill_holes(mask):
    """Fill holes in the alpha mask using scipy morphological operations."""
    try:
        from scipy import ndimage  # type: ignore
        import numpy as np

        filled = ndimage.binary_fill_holes(mask > 0)
        return (filled * 255).astype(mask.dtype)
    except (ImportError, ModuleNotFoundError):
        return mask


def _remove_small_islands(mask, threshold: int):
    """Remove small disconnected regions (islands) from the mask."""
    try:
        from scipy import ndimage  # type: ignore
        import numpy as np

        labeled, num_features = ndimage.label(mask > 0)
        if num_features == 0:
            return mask
        sizes = ndimage.sum(mask > 0, labeled, range(1, num_features + 1))
        significant = np.zeros_like(mask, dtype=bool)
        for i, size in enumerate(sizes):
            if size >= threshold:
                significant |= labeled == (i + 1)
        return (significant * 255).astype(mask.dtype)
    except (ImportError, ModuleNotFoundError):
        return mask


def _tight_crop_and_square_pad(arr, padding: int):
    """Crop to the bounding box of non-transparent pixels and pad to square."""
    import numpy as np

    mask = arr[:, :, 3] > 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return arr

    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    h, w = arr.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)

    cropped = arr[rmin : rmax + 1, cmin : cmax + 1]
    ch, cw = cropped.shape[:2]

    size = max(ch, cw)
    if ch == cw:
        return cropped

    padded = np.zeros((size, size, 4), dtype=arr.dtype)
    row_offset = (size - ch) // 2
    col_offset = (size - cw) // 2
    padded[row_offset : row_offset + ch, col_offset : col_offset + cw] = cropped
    return padded
