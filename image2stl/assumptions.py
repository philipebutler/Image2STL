from __future__ import annotations

from pathlib import Path
from typing import Any


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _resolve_assumption_policy(preset: str) -> dict:
    """Resolve preset into correction strengths and safety limits."""
    preset_key = (preset or "standard").strip().lower()
    table = {
        "conservative": {
            "preset": "conservative",
            "min_confidence": 0.85,
            "flat_bottom_strength": 0.85,
            "symmetry_strength": 0.20,
            "max_volume_delta": 0.04,
        },
        "standard": {
            "preset": "standard",
            "min_confidence": 0.75,
            "flat_bottom_strength": 1.00,
            "symmetry_strength": 0.35,
            "max_volume_delta": 0.08,
        },
        "aggressive": {
            "preset": "aggressive",
            "min_confidence": 0.60,
            "flat_bottom_strength": 1.00,
            "symmetry_strength": 0.50,
            "max_volume_delta": 0.12,
        },
    }
    return table.get(preset_key, table["standard"])


def _estimate_bottom_visibility_confidence(image_paths: list[str]) -> float:
    """Estimate confidence that underside is missing from captures.

    Heuristic: if foreground touches the bottom edge in many images, the object
    likely sits on a surface and underside views are missing.
    """
    if not image_paths:
        return 0.0

    try:
        from PIL import Image
        import numpy as np  # type: ignore
    except (ImportError, ModuleNotFoundError):
        return 0.0

    touches_bottom = 0
    usable = 0
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                rgba = np.array(img.convert("RGBA"))
                alpha = rgba[:, :, 3]
                mask = alpha > 20
                if not mask.any():
                    continue
                usable += 1
                ys = np.where(mask.any(axis=1))[0]
                if ys.size == 0:
                    continue
                lowest = int(ys[-1])
                height = alpha.shape[0]
                if lowest >= int(height * 0.97):
                    touches_bottom += 1
        except (OSError, ValueError):
            continue

    if usable == 0:
        return 0.0
    return _clamp(touches_bottom / usable)


def _flat_bottom_confidence(vertices, image_paths: list[str]) -> float:
    z_vals = vertices[:, 2]
    z_min = float(z_vals.min())
    z_max = float(z_vals.max())
    z_range = max(z_max - z_min, 1e-6)

    band_tol = max(z_range * 0.015, 1e-5)
    bottom_band = z_vals <= (z_min + band_tol)
    if not bottom_band.any():
        return 0.0

    flatness = 1.0 - min(1.0, float(z_vals[bottom_band].std()) / max(band_tol, 1e-6))
    band_ratio = float(bottom_band.mean())
    band_score = _clamp(band_ratio / 0.08)
    underside_missing = _estimate_bottom_visibility_confidence(image_paths)

    confidence = (0.45 * flatness) + (0.25 * band_score) + (0.30 * underside_missing)
    return _clamp(confidence)


def _apply_flat_bottom(vertices, strength: float = 1.0):
    strength = _clamp(strength)
    z_vals = vertices[:, 2]
    z_min = float(z_vals.min())
    z_max = float(z_vals.max())
    z_range = max(z_max - z_min, 1e-6)
    flatten_band = z_vals <= (z_min + max(z_range * 0.025, 1e-5))
    if not flatten_band.any():
        return vertices, 0

    updated = vertices.copy()
    updated_z = updated[flatten_band, 2]
    updated[flatten_band, 2] = (updated_z * (1.0 - strength)) + (z_min * strength)
    return updated, int(flatten_band.sum())


def _symmetry_confidence(vertices) -> tuple[int, float]:
    """Return best axis (0 for X, 1 for Y) and confidence score."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extents = maxs - mins

    best_axis = 0
    best_score = 0.0
    sample = vertices[:: max(1, len(vertices) // 400)]

    for axis in (0, 1):
        center = (mins[axis] + maxs[axis]) / 2.0
        mirrored = sample.copy()
        mirrored[:, axis] = 2.0 * center - mirrored[:, axis]

        # Approximate nearest-neighbor residual using sampled set.
        dists = []
        base = sample[:, [0, 1, 2]]
        for point in mirrored:
            delta = base - point
            dist2 = (delta * delta).sum(axis=1)
            dists.append(float(dist2.min() ** 0.5))
        mean_residual = float(sum(dists) / max(len(dists), 1))
        normalized = mean_residual / max(float(extents.max()), 1e-6)
        score = _clamp(1.0 - (normalized / 0.12))
        if score > best_score:
            best_score = score
            best_axis = axis

    return best_axis, best_score


def _apply_symmetry_correction(vertices, axis: int, strength: float = 0.35):
    """Apply a conservative bilateral symmetry correction on one axis.

    The correction blends each vertex axis-coordinate toward a mirrored
    nearest-neighbour counterpart. This avoids hard snapping and limits
    deformation risk.
    """
    import numpy as np  # type: ignore

    if len(vertices) < 8:
        return vertices, 0

    strength = _clamp(strength)
    if strength <= 0.0:
        return vertices, 0

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins[axis] + maxs[axis]) / 2.0

    n_vertices = len(vertices)
    sample_size = min(1600, n_vertices)
    step = max(1, n_vertices // sample_size)
    sample = vertices[::step]
    if len(sample) == 0:
        return vertices, 0

    mirrored = vertices.copy()
    mirrored[:, axis] = 2.0 * center - mirrored[:, axis]

    updated = vertices.copy()
    chunk_size = 256
    for start in range(0, n_vertices, chunk_size):
        chunk = mirrored[start : start + chunk_size]
        # Distance from mirrored chunk points to sampled original points.
        # Shape: [chunk, sample, xyz]
        delta = sample[None, :, :] - chunk[:, None, :]
        dist2 = (delta * delta).sum(axis=2)
        nearest_idx = np.argmin(dist2, axis=1)
        partner_axis = sample[nearest_idx, axis]

        original_axis = updated[start : start + chunk_size, axis]
        desired_axis = 0.5 * (original_axis + (2.0 * center - partner_axis))
        updated[start : start + chunk_size, axis] = (
            original_axis * (1.0 - strength) + desired_axis * strength
        )

    moved = int(np.count_nonzero(np.abs(updated[:, axis] - vertices[:, axis]) > 1e-8))
    return updated, moved


def apply_mesh_assumptions(
    mesh_path: Path,
    *,
    image_paths: list[str],
    flat_bottom_enabled: bool,
    symmetry_enabled: bool,
    confidence_threshold: float,
    preset: str = "standard",
) -> dict:
    """Apply assumption-guided mesh corrections with conservative rollback.

    Returns a structured report and modifies mesh_path in place only when a
    correction passes confidence and safety gates.
    """
    applied: list[str] = []
    skipped: list[str] = []
    confidence: dict[str, float] = {}
    policy = _resolve_assumption_policy(preset)
    effective_confidence_threshold = max(
        _clamp(confidence_threshold),
        float(policy["min_confidence"]),
    )

    def _report(extra: dict | None = None) -> dict:
        payload = {
            "enabled": True,
            "applied": applied,
            "skipped": skipped,
            "confidence": confidence,
            "appliedCount": len(applied),
            "preset": policy["preset"],
            "effectiveConfidenceThreshold": effective_confidence_threshold,
        }
        if extra:
            payload.update(extra)
        return payload

    try:
        import numpy as np  # type: ignore
        import trimesh  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        skipped.append("missing_mesh_dependencies")
        return _report({"detail": str(exc)})

    try:
        loaded = trimesh.load(str(mesh_path), force="mesh")
        mesh: Any
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.dump(concatenate=True)
        elif isinstance(loaded, (list, tuple)):
            mesh = loaded[0] if loaded else None
        else:
            mesh = loaded
    except (OSError, RuntimeError, ValueError) as exc:
        skipped.append("mesh_load_failed")
        return _report({"detail": str(exc)})

    if mesh is None:
        skipped.append("empty_mesh")
        return _report()

    vertices = np.array(getattr(mesh, "vertices", []), dtype=float)
    if vertices.size == 0:
        skipped.append("empty_mesh")
        return _report()

    original_vertices = vertices.copy()
    original_volume = float(getattr(mesh, "volume", 0.0)) if getattr(mesh, "is_watertight", False) else None
    flat_bottom_modified_vertices = 0
    symmetry_axis = None
    symmetry_modified_vertices = 0

    if flat_bottom_enabled:
        fb_conf = _flat_bottom_confidence(vertices, image_paths)
        confidence["flat_bottom"] = fb_conf
        if fb_conf >= effective_confidence_threshold:
            new_vertices, modified = _apply_flat_bottom(
                vertices,
                strength=float(policy["flat_bottom_strength"]),
            )
            if modified > 0:
                vertices = new_vertices
                applied.append("flat_bottom")
                flat_bottom_modified_vertices = modified
        else:
            skipped.append("flat_bottom_low_confidence")
    if symmetry_enabled:
        axis, sym_conf = _symmetry_confidence(vertices)
        confidence["symmetry"] = sym_conf
        if sym_conf >= effective_confidence_threshold:
            corrected, moved = _apply_symmetry_correction(
                vertices,
                axis=axis,
                strength=float(policy["symmetry_strength"]),
            )
            symmetry_axis = "x" if axis == 0 else "y"
            if moved > 0:
                vertices = corrected
                applied.append("symmetry")
                symmetry_modified_vertices = moved
            else:
                skipped.append("symmetry_noop")
        else:
            skipped.append("symmetry_low_confidence")
            symmetry_axis = None
            symmetry_modified_vertices = 0
    else:
        symmetry_modified_vertices = 0
    if not applied:
        extra: dict = {}
        if flat_bottom_modified_vertices:
            extra["flatBottomModifiedVertices"] = flat_bottom_modified_vertices
        if symmetry_modified_vertices:
            extra["symmetryModifiedVertices"] = symmetry_modified_vertices
        if symmetry_axis:
            extra["symmetryAxis"] = symmetry_axis
        return _report(extra)

    # Safety gate: reject changes if watertight volume drifts too much.
    mesh.vertices = vertices
    try:
        mesh.fix_normals()
    except Exception:
        pass
    if original_volume is not None and getattr(mesh, "is_watertight", False):
        new_volume = float(getattr(mesh, "volume", 0.0))
        denominator = max(abs(original_volume), 1e-6)
        delta_ratio = abs(new_volume - original_volume) / denominator
        volume_delta_ratio = delta_ratio
        if delta_ratio > float(policy["max_volume_delta"]):
            mesh.vertices = original_vertices
            skipped.append("rollback_volume_delta")
            applied.clear()
            mesh.export(str(mesh_path))
            return _report({"volumeDeltaRatio": volume_delta_ratio})
    else:
        volume_delta_ratio = None

    mesh.export(str(mesh_path))
    extra = {
        "flatBottomModifiedVertices": flat_bottom_modified_vertices,
    }
    if symmetry_modified_vertices:
        extra["symmetryModifiedVertices"] = symmetry_modified_vertices
    if symmetry_axis:
        extra["symmetryAxis"] = symmetry_axis
    if volume_delta_ratio is not None:
        extra["volumeDeltaRatio"] = volume_delta_ratio
    return _report(extra)
