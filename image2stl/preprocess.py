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
    edge_feather_radius: int = 0,
    contrast_strength: float = 0.0,
    crop_mode: str = "square",
    consistency_strength: float = 0.5,
    denoise_strength: float = 0.2,
    deblur_strength: float = 0.2,
    background_mode: str = "transparent",
    background_color: str = "#FFFFFF",
) -> str:
    """Compute a short hash of preprocessing parameters for deterministic naming."""
    key = (
        f"{strength:.3f}|{hole_fill}|{island_threshold}|{crop_padding}"
        f"|{edge_feather_radius}|{contrast_strength:.2f}|{crop_mode}"
        f"|{consistency_strength:.2f}|{denoise_strength:.2f}|{deblur_strength:.2f}"
        f"|{background_mode}|{background_color.upper()}"
    )
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
    min_output_size: int = 512,
    edge_feather_radius: int = 2,
    contrast_strength: float = 0.0,
    crop_mode: str = "square",
    consistency_strength: float = 0.5,
    denoise_strength: float = 0.2,
    deblur_strength: float = 0.2,
    background_mode: str = "transparent",
    background_color: str = "#FFFFFF",
    consistency_reference: dict | None = None,
) -> Path:
    """Isolate the foreground in an image and save the result as RGBA PNG.

    Args:
        source_path: Path to source image.
        output_dir: Directory to write processed images.
        strength: Background removal strength (controls post-processing aggressiveness).
        hole_fill: Whether to fill holes in the mask.
        island_removal_threshold: Minimum connected component size to keep.
        crop_padding: Padding (in pixels) to add around the tight crop.
        min_output_size: Minimum output image dimension in pixels.  If the
            processed image is smaller than this, it is upscaled using
            high-quality (LANCZOS) resampling so the result is never pixelated.
        edge_feather_radius: Radius in pixels for alpha edge feathering to
            smooth mask boundaries (0 = disabled).
        contrast_strength: Foreground contrast/sharpness enhancement level
            (0.0 = none, 1.0 = maximum).
        crop_mode: Crop mode ("square" or "original").
        consistency_strength: Cross-image consistency strength (0.0–1.0).
        denoise_strength: Foreground denoise strength (0.0–1.0).
        deblur_strength: Foreground deblur/sharpen strength (0.0–1.0).
        background_mode: Output background mode ("transparent" or "solid").
        background_color: Solid background color in #RRGGBB format.
        consistency_reference: Optional shared reference stats for cross-image
            consistency across a batch.

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
        strength, hole_fill, island_removal_threshold, crop_padding,
        edge_feather_radius, contrast_strength, crop_mode,
        consistency_strength, denoise_strength, deblur_strength,
        background_mode, background_color,
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

    rgba = _postprocess_mask(
        rgba, hole_fill, island_removal_threshold, crop_padding,
        min_output_size, edge_feather_radius, contrast_strength,
        crop_mode=crop_mode,
        consistency_strength=consistency_strength,
        denoise_strength=denoise_strength,
        deblur_strength=deblur_strength,
        background_mode=background_mode,
        background_color=background_color,
        consistency_reference=consistency_reference,
    )
    rgba.save(str(output_path), format="PNG")
    return output_path


def _postprocess_mask(
    rgba_image,
    hole_fill: bool,
    island_threshold: int,
    crop_padding: int,
    min_output_size: int = 512,
    edge_feather_radius: int = 2,
    contrast_strength: float = 0.0,
    crop_mode: str = "square",
    consistency_strength: float = 0.5,
    denoise_strength: float = 0.2,
    deblur_strength: float = 0.2,
    background_mode: str = "transparent",
    background_color: str = "#FFFFFF",
    consistency_reference: dict | None = None,
):
    """Apply mask cleanup, edge refinement, feathering, contrast enhancement,
    tight crop, square pad, and minimum-size upscale to an RGBA image."""
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

    # Morphological edge refinement: close then open to smooth jagged edges
    cleanup_mask = _refine_mask_edges(cleanup_mask)

    original_alpha = _smooth_alpha_channel(original_alpha)
    keep_region = cleanup_mask > 0
    arr[:, :, 3] = np.where(keep_region, original_alpha, 0).astype(original_alpha.dtype)

    # Edge feathering: soften alpha transitions at mask boundaries
    if edge_feather_radius > 0:
        arr[:, :, 3] = _feather_edges(arr[:, :, 3], edge_feather_radius)

    # Contrast/sharpness enhancement on foreground RGB channels
    if contrast_strength > 0.0:
        arr = _enhance_foreground(arr, contrast_strength)

    arr = _apply_cross_image_consistency(arr, consistency_reference, consistency_strength)
    arr = _denoise_foreground(arr, denoise_strength)
    if deblur_strength > 0.0:
        arr = _enhance_foreground(arr, deblur_strength)

    arr = _crop_and_pad(arr, crop_padding, crop_mode)
    arr = _apply_background_mode(arr, background_mode, background_color)
    result = Image.fromarray(arr, mode="RGBA")

    # Upscale to min_output_size if the result is too small so that the
    # processed image is never pixelated in the UI or when fed to the
    # reconstruction model.
    if min_output_size > 0:
        w, h = result.size
        if w < min_output_size or h < min_output_size:
            scale = max(min_output_size / max(w, 1), min_output_size / max(h, 1))
            new_w = max(min_output_size, int(round(w * scale)))
            new_h = max(min_output_size, int(round(h * scale)))
            result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return result


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


def _refine_mask_edges(mask):
    """Smooth jagged mask edges using morphological close then open."""
    try:
        from scipy import ndimage  # type: ignore
        import numpy as np

        struct = ndimage.generate_binary_structure(2, 1)
        # Close: fill small gaps along edges
        refined = ndimage.binary_closing(mask > 0, structure=struct, iterations=2)
        # Open: remove small protrusions along edges
        refined = ndimage.binary_opening(refined, structure=struct, iterations=1)
        return (refined * 255).astype(mask.dtype)
    except (ImportError, ModuleNotFoundError):
        return mask


def _feather_edges(alpha_channel, radius: int):
    """Apply Gaussian feathering to alpha channel edges for smoother transitions."""
    try:
        from scipy import ndimage  # type: ignore
        import numpy as np

        # Only blur near edges: find boundary pixels first
        binary = alpha_channel > 0
        eroded = ndimage.binary_erosion(binary, iterations=max(1, radius))
        dilated = ndimage.binary_dilation(binary, iterations=max(1, radius))
        edge_band = dilated & ~eroded

        # Feather sigma is ~80% of the pixel radius for a natural falloff
        sigma = max(0.5, radius * 0.8)
        blurred = ndimage.gaussian_filter(alpha_channel.astype(float), sigma=sigma)
        result = alpha_channel.copy().astype(float)
        result[edge_band] = blurred[edge_band]
        return result.clip(0, 255).astype(alpha_channel.dtype)
    except (ImportError, ModuleNotFoundError):
        try:
            import numpy as np
            from PIL import Image, ImageFilter

            pil_alpha = Image.fromarray(alpha_channel, mode="L")
            blurred = pil_alpha.filter(ImageFilter.GaussianBlur(radius=max(1, radius)))
            return np.array(blurred).astype(alpha_channel.dtype)
        except (ImportError, ModuleNotFoundError):
            return alpha_channel


def _enhance_foreground(arr, strength: float):
    """Enhance contrast and sharpness of foreground RGB channels.

    Uses unsharp masking on the RGB channels where the alpha channel
    indicates foreground, improving surface detail for reconstruction.
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter
    except (ImportError, ModuleNotFoundError):
        return arr

    strength = max(0.0, min(1.0, float(strength)))
    alpha = arr[:, :, 3]
    fg_mask = alpha > 0

    if not fg_mask.any():
        return arr

    rgb = arr[:, :, :3].copy()
    pil_rgb = Image.fromarray(rgb, mode="RGB")

    # Unsharp mask with strength-scaled parameters:
    #   radius:    1.5 (subtle) to 3.0 (aggressive) px
    #   percent:   80% (subtle) to 200% (aggressive) sharpening
    #   threshold: 4 (subtle, skips low-contrast edges) to 1 (aggressive)
    usm_radius = 1.5 + strength * 1.5
    usm_percent = int(80 + strength * 120)
    usm_threshold = max(1, int(4 - strength * 3))
    sharpened = pil_rgb.filter(
        ImageFilter.UnsharpMask(radius=usm_radius, percent=usm_percent, threshold=usm_threshold)
    )
    sharp_arr = np.array(sharpened)

    # Only apply enhancement to foreground pixels
    for c in range(3):
        arr[:, :, c] = np.where(fg_mask, sharp_arr[:, :, c], arr[:, :, c])

    return arr


def build_consistency_reference(source_path: Path) -> dict | None:
    """Build reference image statistics used for cross-image consistency."""
    try:
        import numpy as np
        from PIL import Image
    except (ImportError, ModuleNotFoundError):
        return None

    with Image.open(source_path) as img:
        rgb = np.array(img.convert("RGB")).astype(np.float32)
    return {"mean": rgb.mean(axis=(0, 1)), "std": rgb.std(axis=(0, 1)) + 1e-6}


def _apply_cross_image_consistency(arr, reference: dict | None, strength: float):
    """Apply light color normalization toward a shared reference image profile."""
    if not reference or strength <= 0.0:
        return arr
    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError):
        return arr

    strength = max(0.0, min(1.0, float(strength)))
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3] > 0
    if not alpha.any():
        return arr

    fg = rgb[alpha]
    current_mean = fg.mean(axis=0)
    current_std = fg.std(axis=0) + 1e-6
    ref_mean = np.array(reference.get("mean", current_mean), dtype=np.float32)
    ref_std = np.array(reference.get("std", current_std), dtype=np.float32)

    normalized = ((fg - current_mean) / current_std) * ref_std + ref_mean
    blended = fg * (1.0 - strength) + normalized * strength
    rgb[alpha] = blended
    arr[:, :, :3] = rgb.clip(0, 255).astype(arr.dtype)
    return arr


def _denoise_foreground(arr, strength: float):
    """Reduce foreground grain while preserving boundaries."""
    if strength <= 0.0:
        return arr
    try:
        import numpy as np
        from PIL import Image, ImageFilter
    except (ImportError, ModuleNotFoundError):
        return arr

    strength = max(0.0, min(1.0, float(strength)))
    radius = 0.5 + strength * 1.5
    alpha = arr[:, :, 3] > 0
    if not alpha.any():
        return arr

    rgb = arr[:, :, :3]
    smoothed = np.array(
        Image.fromarray(rgb, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    )
    for c in range(3):
        arr[:, :, c] = np.where(alpha, smoothed[:, :, c], arr[:, :, c])
    return arr


def _parse_hex_color(color: str) -> tuple[int, int, int]:
    value = (color or "#FFFFFF").strip()
    if len(value) == 7 and value.startswith("#"):
        try:
            return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)
        except ValueError:
            pass
    return (255, 255, 255)


def _apply_background_mode(arr, background_mode: str, background_color: str):
    """Optionally flatten transparent background to a solid color."""
    if (background_mode or "transparent").lower() != "solid":
        return arr
    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError):
        return arr

    bg_r, bg_g, bg_b = _parse_hex_color(background_color)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    inv_alpha = 1.0 - alpha

    for channel, bg in enumerate((bg_r, bg_g, bg_b)):
        fg = arr[:, :, channel].astype(np.float32)
        arr[:, :, channel] = (fg * alpha + bg * inv_alpha).clip(0, 255).astype(arr.dtype)
    arr[:, :, 3] = 255
    return arr


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


def _crop_and_pad(arr, padding: int, crop_mode: str = "square"):
    """Crop to foreground bounds and optionally pad to square framing."""
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

    if (crop_mode or "square").lower() == "original":
        return cropped

    size = max(ch, cw)
    if ch == cw:
        return cropped

    padded = np.zeros((size, size, 4), dtype=arr.dtype)
    row_offset = (size - ch) // 2
    col_offset = (size - cw) // 2
    padded[row_offset : row_offset + ch, col_offset : col_offset + cw] = cropped
    return padded
