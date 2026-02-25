"""Foreground isolation preprocessing using rembg."""
from __future__ import annotations

import hashlib
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
    mask = arr[:, :, 3].copy()

    if hole_fill:
        mask = _fill_holes(mask)
    if island_threshold > 0:
        mask = _remove_small_islands(mask, island_threshold)

    arr[:, :, 3] = mask
    arr = _tight_crop_and_square_pad(arr, crop_padding)
    return Image.fromarray(arr, mode="RGBA")


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
