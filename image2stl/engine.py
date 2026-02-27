from __future__ import annotations

import json
import os
import importlib
import importlib.util
import importlib.metadata
import threading
import time
import contextvars
import sys
import inspect
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from .errors import SUPPORTED_IMAGE_EXTENSIONS, make_error

_HEIF_REGISTERED = False
_AVIF_REGISTERED = False

DEFAULT_TARGET_FACE_COUNT = 100000
MIN_IMAGE_DIMENSION = 512


def _ensure_heif_support() -> None:
    """Register pillow-heif opener so Pillow can read HEIC/HEIF images."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_REGISTERED = True
    except (ImportError, ModuleNotFoundError):
        pass


def _ensure_avif_support() -> None:
    """Register pillow-avif-plugin so Pillow can read AVIF images."""
    global _AVIF_REGISTERED
    if _AVIF_REGISTERED:
        return
    try:
        import pillow_avif  # noqa: F401

        _AVIF_REGISTERED = True
    except (ImportError, ModuleNotFoundError):
        pass


WARNING_THRESHOLD_SECONDS = 600
DEFAULT_INPUT_DIMENSIONS_MM = (100.0, 120.0, 80.0)
MESHY_BASE_URL = "https://api.meshy.ai/v1"
MESHY_API_KEY_ENV = "MESHY_API_KEY"
MESHY_TIMEOUT_SECONDS = 600
_CURRENT_OPERATION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_operation_id", default=None)
_CANCELLED_OPERATION_IDS: set[str] = set()
_CANCEL_LOCK = threading.Lock()


class OperationCancelledError(RuntimeError):
    pass


class MissingDependenciesError(RuntimeError):
    def __init__(self, missing: list[dict]):
        self.missing = missing
        names = ", ".join(item["module"] for item in missing)
        super().__init__(f"Missing Python dependencies: {names}")


class ModelWeightsUnavailableError(RuntimeError):
    pass


LOCAL_DEPENDENCY_SPEC = [
    {"module": "torch", "package": "torch", "required": True},
    {"module": "PIL", "package": "Pillow", "required": True},
    {
        "module": "tsr",
        "package": "TripoSR",
        "required": True,
        "installTarget": "__TRIPOSR_SOURCE_CHECKOUT__",
    },
    {"module": "transformers", "package": "transformers", "required": True},
    {"module": "huggingface_hub", "package": "huggingface-hub", "required": True},
    {"module": "numpy", "package": "numpy", "required": False},
    {"module": "pillow_heif", "package": "pillow-heif", "required": False},
    {"module": "pillow_avif", "package": "pillow-avif-plugin", "required": False},
]


def _get_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_local_dependency_report() -> dict:
    dependencies: list[dict] = []
    missing: list[dict] = []
    all_required_available = True

    for spec in LOCAL_DEPENDENCY_SPEC:
        module_name = spec["module"]
        package_name = spec["package"]
        install_target = spec.get("installTarget", package_name)
        required = bool(spec["required"])
        module_spec = importlib.util.find_spec(module_name)
        installed = module_spec is not None
        version = _get_package_version(package_name) if installed else None
        dependency = {
            "module": module_name,
            "package": package_name,
            "installTarget": install_target,
            "required": required,
            "installed": installed,
            "version": version,
        }
        dependencies.append(dependency)
        if required and not installed:
            all_required_available = False
            missing.append({"module": module_name, "package": package_name, "installTarget": install_target})

    return {
        "dependencies": dependencies,
        "missing": missing,
        "ok": all_required_available,
    }


def _ensure_local_dependencies() -> dict:
    report = _collect_local_dependency_report()
    if not report["ok"]:
        raise MissingDependenciesError(report["missing"])
    return report


def _detect_triposr_cache_state(repo_id: str = "stabilityai/TripoSR") -> dict:
    state = {
        "repoId": repo_id,
        "cacheCheckSupported": False,
        "cacheStatusBeforeLoad": "unknown",
        "downloadLikelyRequired": None,
    }
    try:
        from huggingface_hub import hf_hub_download
    except (ImportError, ModuleNotFoundError):
        return state

    state["cacheCheckSupported"] = True
    try:
        hf_hub_download(repo_id=repo_id, filename="config.yaml", local_files_only=True)
        hf_hub_download(repo_id=repo_id, filename="model.ckpt", local_files_only=True)
        state["cacheStatusBeforeLoad"] = "cached"
        state["downloadLikelyRequired"] = False
    except Exception as exc:
        detail = str(exc).lower()
        not_cached_markers = (
            "local files only",
            "not found in local cache",
            "cannot find the requested files",
            "cannot find the requested file",
            "no such file",
        )
        if any(marker in detail for marker in not_cached_markers):
            state["cacheStatusBeforeLoad"] = "not_cached"
            state["downloadLikelyRequired"] = True
        else:
            state["cacheStatusBeforeLoad"] = "unknown"
            state["downloadLikelyRequired"] = None
            state["cacheCheckError"] = str(exc)

    return state


def parse_json_line(line: str) -> dict:
    parsed = json.loads(line)
    if not isinstance(parsed, dict):
        raise ValueError("Command payload must be a JSON object")
    return parsed


def calculate_scale_factor(current_dimensions_mm: tuple[float, float, float], target_size_mm: float, axis: str) -> float:
    width, height, depth = current_dimensions_mm
    axis_map = {
        "width": width,
        "height": height,
        "depth": depth,
        "longest": max(current_dimensions_mm),
    }
    current = axis_map.get(axis)
    if current is None or current <= 0:
        raise ValueError("Invalid axis or dimensions")
    return target_size_mm / current


def _status(progress: float, text: str, estimated: int | None = None) -> dict:
    msg = {"type": "progress", "progress": progress, "status": text}
    if estimated is not None:
        msg["estimatedSecondsRemaining"] = estimated
        if estimated > WARNING_THRESHOLD_SECONDS:
            msg["warning"] = "Estimated processing time exceeds 10 minutes"
    return msg


def _validate_reconstruction(command: dict) -> dict | None:
    images = command.get("images", [])
    if len(images) < 3:
        return make_error("reconstruct", "INSUFFICIENT_IMAGES")
    if len(images) > 50:
        return make_error("reconstruct", "TOO_MANY_IMAGES")
    for image in images:
        if Path(image).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return make_error("reconstruct", "UNSUPPORTED_FILE_FORMAT")
    if command.get("mode") == "cloud" and command.get("simulateApiError"):
        return make_error("reconstruct", "API_ERROR")
    return None


def _mark_operation_cancelled(operation_id: str | None) -> None:
    if not operation_id:
        return
    with _CANCEL_LOCK:
        _CANCELLED_OPERATION_IDS.add(operation_id)


def _is_operation_cancelled(operation_id: str | None = None) -> bool:
    op_id = operation_id or _CURRENT_OPERATION_ID.get()
    if not op_id:
        return False
    with _CANCEL_LOCK:
        return op_id in _CANCELLED_OPERATION_IDS


def _clear_operation_cancelled(operation_id: str | None) -> None:
    if not operation_id:
        return
    with _CANCEL_LOCK:
        _CANCELLED_OPERATION_IDS.discard(operation_id)


def _cleanup_operation(token: contextvars.Token[str | None], operation_id: str | None) -> None:
    _CURRENT_OPERATION_ID.reset(token)
    _clear_operation_cancelled(operation_id)


def check_image_quality(image_paths: list[str]) -> list[dict]:
    """Pre-flight image quality checks: resolution and blur detection.

    Returns a list of warning dicts for images that may cause problems during
    reconstruction.  Each dict contains *path*, *issue*, and *suggestion* keys.
    """
    warnings: list[dict] = []
    try:
        from PIL import Image
    except (ImportError, ModuleNotFoundError):
        return warnings

    _ensure_heif_support()
    _ensure_avif_support()

    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                    warnings.append({
                        "path": image_path,
                        "issue": "low_resolution",
                        "detail": f"Image is {w}x{h}; minimum recommended is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}",
                        "suggestion": "Use higher resolution images for better reconstruction quality",
                    })
                # Blur detection via Laplacian variance (requires numpy)
                try:
                    import numpy as np  # type: ignore

                    grey = img.convert("L")
                    arr = np.array(grey, dtype=np.float64)
                    # Discrete Laplacian: sum of 4-connected neighbours minus 4× centre pixel.
                    # Low variance of this kernel indicates a blurry image.
                    laplacian = (
                        arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4 * arr[1:-1, 1:-1]
                    )
                    variance = float(np.var(laplacian))
                    if variance < 100.0:
                        warnings.append({
                            "path": image_path,
                            "issue": "blurry",
                            "detail": f"Laplacian variance {variance:.1f} is below threshold 100",
                            "suggestion": "Use sharper, well-focused images",
                        })
                except (ImportError, ModuleNotFoundError):
                    pass
        except (OSError, ValueError):
            warnings.append({
                "path": image_path,
                "issue": "unreadable",
                "detail": "Could not open image file",
                "suggestion": "Ensure the file is a valid image",
            })
    return warnings


def _run_triposr_local(images: list[str], target: Path) -> dict:
    """Run TripoSR with the primary capture image; additional captures are reserved for future multi-view tuning."""
    dependency_report = _ensure_local_dependencies()
    try:
        import torch
        from PIL import Image
        from tsr.system import TSR
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - dependency availability is environment-specific
        raise RuntimeError("TripoSR dependencies unavailable") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_state = _detect_triposr_cache_state()
    if _is_operation_cancelled():
        raise OperationCancelledError()
    try:
        from_pretrained_params = inspect.signature(TSR.from_pretrained).parameters
        if "config_name" in from_pretrained_params and "weight_name" in from_pretrained_params:
            model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
        else:
            model = TSR.from_pretrained("stabilityai/TripoSR")
    except Exception as exc:
        raise ModelWeightsUnavailableError(f"Unable to load TripoSR model weights: {exc}") from exc
    if hasattr(model, "to"):
        model = model.to(device)
    primary_image = images[0]  # TripoSR inference currently runs from the primary capture image.
    _ensure_heif_support()
    _ensure_avif_support()
    with Image.open(primary_image) as src_image:
        image = src_image.convert("RGB")
        scene_codes = model([image], device=device)
    extract_mesh_params = inspect.signature(model.extract_mesh).parameters
    if "has_vertex_color" in extract_mesh_params:
        meshes = model.extract_mesh(scene_codes, has_vertex_color=False)
    else:
        meshes = model.extract_mesh(scene_codes)
    if _is_operation_cancelled():
        raise OperationCancelledError()
    mesh = meshes[0]
    mesh.export(str(target))
    return {
        "vertices": len(getattr(mesh, "vertices", [])),
        "faces": len(getattr(mesh, "faces", [])),
        "runtime": {
            "device": device,
            "dependenciesOk": dependency_report["ok"],
        },
        "model": {
            **cache_state,
            "repoId": "stabilityai/TripoSR",
        },
    }


def _read_meshy_api_key(command: dict) -> str | None:
    if command.get("apiKey"):
        return str(command["apiKey"])
    env_var = command.get("apiKeyEnvVar", MESHY_API_KEY_ENV)
    return os.environ.get(env_var)


def _run_meshy_cloud(images: list[str], target: Path, api_key: str) -> dict:
    if not all(isinstance(image, str) and image.strip() for image in images):
        raise ValueError("Meshy image inputs must be non-empty strings")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = json.dumps({"mode": "image-to-3d", "images": images}).encode("utf-8")
    create_request = urlrequest.Request(
        f"{MESHY_BASE_URL}/image-to-3d",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urlrequest.urlopen(create_request, timeout=MESHY_TIMEOUT_SECONDS) as response:
        task = json.loads(response.read().decode("utf-8"))
    task_id = task.get("id")
    if not task_id:
        raise RuntimeError("Meshy task creation failed")
    timeout_seconds = MESHY_TIMEOUT_SECONDS
    status_url = f"{MESHY_BASE_URL}/tasks/{task_id}"
    deadline = time.monotonic() + timeout_seconds
    final_task: dict = {}
    while time.monotonic() < deadline:
        if _is_operation_cancelled():
            raise OperationCancelledError()
        status_request = urlrequest.Request(status_url, headers=headers, method="GET")
        with urlrequest.urlopen(status_request, timeout=timeout_seconds) as response:
            final_task = json.loads(response.read().decode("utf-8"))
        status = final_task.get("status", "").lower()
        if status in {"completed", "succeeded"}:
            break
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError("Meshy task failed")
        time.sleep(5)
    if final_task.get("status", "").lower() not in {"completed", "succeeded"}:
        raise RuntimeError("Meshy task timed out")
    model_url = final_task.get("model_url") or final_task.get("modelUrl")
    if not model_url:
        raise RuntimeError("Meshy did not return a model URL")
    with urlrequest.urlopen(model_url, timeout=600) as response:
        target.write_bytes(response.read())
    return {
        "vertices": int(final_task.get("vertices", 0)),
        "faces": int(final_task.get("faces", 0)),
    }


def process_command(command: dict) -> list[dict]:
    cmd = command.get("command")
    operation_id = command.get("operationId")
    if cmd == "cancel":
        cancel_target = str(operation_id or command.get("targetOperationId") or "")
        _mark_operation_cancelled(cancel_target)
        return [{"type": "success", "command": "cancel", "operationId": cancel_target}]
    output = []
    if cmd == "check_environment":
        mode = command.get("mode", "local")
        response = {
            "type": "success",
            "command": "check_environment",
            "mode": mode,
            "python": {
                "version": sys.version.split()[0],
                "executable": sys.executable,
            },
        }
        if mode == "local":
            report = _collect_local_dependency_report()
            response["local"] = {
                "ok": report["ok"],
                "dependencies": report["dependencies"],
                "missing": report["missing"],
            }
            response["model"] = _detect_triposr_cache_state()
        elif mode == "cloud":
            response["cloud"] = {
                "apiKeyConfigured": bool(_read_meshy_api_key(command)),
                "apiKeyEnvVar": command.get("apiKeyEnvVar", MESHY_API_KEY_ENV),
            }
        return [response]

    if cmd == "reconstruct":
        validation_error = _validate_reconstruction(command)
        if validation_error:
            return [validation_error]
        mode = command.get("mode", "local")
        op_id = str(operation_id) if operation_id else None
        token = _CURRENT_OPERATION_ID.set(op_id)
        def _cancelled_response() -> list[dict]:
            _cleanup_operation(token, op_id)
            return [make_error("reconstruct", "OPERATION_CANCELLED")]
        output.append(_status(0.1, "Loading images...", estimated=120))
        if _is_operation_cancelled(op_id):
            return _cancelled_response()
        if mode == "local":
            output.append(_status(0.25, "Checking Python dependencies...", estimated=20))
            output.append(_status(0.35, "Checking TripoSR model cache...", estimated=20))
            output.append(_status(0.5, "Loading/downloading TripoSR model weights...", estimated=480))
        else:
            output.append(_status(0.5, "Running cloud reconstruction...", estimated=480))
        if _is_operation_cancelled(op_id):
            return _cancelled_response()
        output.append(_status(0.8, "Repairing mesh...", estimated=90))
        if _is_operation_cancelled(op_id):
            return _cancelled_response()
        output.append(_status(0.95, "Generating preview...", estimated=20))
        target = Path(command["outputPath"])
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mode == "cloud":
                api_key = _read_meshy_api_key(command)
                if not api_key:
                    return [make_error("reconstruct", "API_ERROR")]
                stats = _run_meshy_cloud(command["images"], target, api_key)
            else:
                stats = _run_triposr_local(command["images"], target)
        except OperationCancelledError:
            _cleanup_operation(token, op_id)
            return [make_error("reconstruct", "OPERATION_CANCELLED")]
        except MissingDependenciesError as exc:
            _cleanup_operation(token, op_id)
            error = make_error("reconstruct", "PYTHON_DEPENDENCIES_MISSING")
            error["missingDependencies"] = exc.missing
            return [error]
        except ModelWeightsUnavailableError as exc:
            _cleanup_operation(token, op_id)
            error = make_error("reconstruct", "MODEL_WEIGHTS_UNAVAILABLE")
            error["detail"] = str(exc)
            return [error]
        except (urlerror.URLError, RuntimeError, ValueError) as exc:
            error_code = "API_ERROR" if mode == "cloud" else "RECONSTRUCTION_FAILED"
            _cleanup_operation(token, op_id)
            error = make_error("reconstruct", error_code)
            error["detail"] = str(exc)
            return [error]
        _cleanup_operation(token, op_id)
        output.append(
            {
                "type": "success",
                "command": "reconstruct",
                "outputPath": str(target),
                "stats": stats,
            }
        )
        return output

    if cmd == "repair":
        src = Path(command["inputMesh"])
        dst = Path(command["outputMesh"])
        target_faces = int(command.get("targetFaceCount", DEFAULT_TARGET_FACE_COUNT))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if _is_operation_cancelled(str(operation_id) if operation_id else None):
            return [make_error("repair", "OPERATION_CANCELLED")]
        try:
            repaired = None
            try:
                import pymeshlab  # type: ignore

                mesh_set = pymeshlab.MeshSet()
                mesh_set.load_new_mesh(str(src))
                mesh_set.meshing_repair_non_manifold_edges()
                mesh_set.meshing_repair_non_manifold_vertices()
                mesh_set.meshing_close_holes(maxholesize=1000)
                mesh_set.meshing_remove_duplicate_faces()
                mesh_set.meshing_remove_duplicate_vertices()
                mesh_set.meshing_re_orient_all_faces_coherentely()
                # Decimation to target face count (SPEC FR4)
                current = mesh_set.current_mesh()
                if hasattr(current, "face_number") and current.face_number() > target_faces:
                    mesh_set.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
                # Remove internal geometry when available
                try:
                    mesh_set.compute_selection_by_small_disconnected_components_per_face()
                    mesh_set.meshing_remove_selected_faces()
                except AttributeError:
                    pass
                mesh_set.save_current_mesh(str(dst))
                repaired = True
            except (ImportError, ModuleNotFoundError, AttributeError):
                repaired = False
            try:
                import trimesh  # type: ignore
            except (ImportError, ModuleNotFoundError):
                if repaired:
                    return [{"type": "success", "command": "repair", "outputPath": str(dst)}]
                raise

            loaded = trimesh.load(str(dst if repaired else src), force="mesh")
            mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            mesh.fill_holes()
            # Remove small disconnected components (internal geometry)
            if hasattr(mesh, "split") and callable(mesh.split):
                components = mesh.split()
                if len(components) > 1:
                    mesh = max(components, key=lambda c: len(getattr(c, "faces", [])))
            mesh.export(str(dst))
            is_watertight = bool(getattr(mesh, "is_watertight", False))
            return [
                {
                    "type": "success",
                    "command": "repair",
                    "outputPath": str(dst),
                    "stats": {
                        "vertices": len(getattr(mesh, "vertices", [])),
                        "faces": len(getattr(mesh, "faces", [])),
                        "watertight": is_watertight,
                    },
                    "validation": {
                        "watertight": is_watertight,
                        "result": "pass" if is_watertight else "fail",
                        "detail": "Mesh is watertight and printable" if is_watertight else "Mesh has holes; manual repair may be needed",
                    },
                }
            ]
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
            return [make_error("repair", "MESH_REPAIR_FAILED")]

    if cmd == "scale":
        src = Path(command["inputMesh"])
        dst = Path(command["outputMesh"])
        input_dimensions = tuple(command.get("inputDimensionsMm", DEFAULT_INPUT_DIMENSIONS_MM))
        try:
            import trimesh  # type: ignore

            loaded = trimesh.load(str(src), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                mesh = loaded.dump(concatenate=True)
            elif isinstance(loaded, (list, tuple)):
                mesh = loaded[0] if loaded else None
            else:
                mesh = loaded

            if mesh is None:
                raise ValueError("Input mesh could not be loaded")

            bounding_box = getattr(mesh, "bounding_box", None)
            extents = tuple(float(v) for v in getattr(bounding_box, "extents", []))
            dimensions_for_scale = extents if len(extents) == 3 and all(v > 0 for v in extents) else input_dimensions

            factor = calculate_scale_factor(dimensions_for_scale, float(command["targetSizeMm"]), command.get("axis", "longest"))
            mesh.apply_scale(factor)

            dst.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(dst))
            return [{"type": "success", "command": "scale", "outputPath": str(dst), "scaleFactor": factor}]
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
            return [make_error("scale", "FILE_IO_ERROR")]

    if cmd == "check_images":
        images = command.get("images", [])
        warnings = check_image_quality(images)
        return [{"type": "success", "command": "check_images", "warnings": warnings}]

    if cmd == "preprocess_images":
        images = command.get("images", [])
        output_dir = Path(command.get("outputDir", "preview/processed"))
        strength = float(command.get("strength", 0.5))
        hole_fill = bool(command.get("holeFill", True))
        island_threshold = int(command.get("islandRemovalThreshold", 500))
        crop_padding = int(command.get("cropPadding", 10))
        try:
            from .preprocess import preprocess_image
        except ImportError:
            error = make_error("preprocess_images", "REMBG_UNAVAILABLE")
            error["detail"] = "rembg is not installed. Install it with: pip install rembg"
            return [error]
        processed = []
        warnings: list[dict] = []
        for image in images:
            try:
                out = preprocess_image(
                    Path(image),
                    output_dir,
                    strength=strength,
                    hole_fill=hole_fill,
                    island_removal_threshold=island_threshold,
                    crop_padding=crop_padding,
                )
                processed.append(str(out))
            except ImportError:
                error = make_error("preprocess_images", "REMBG_UNAVAILABLE")
                error["detail"] = "rembg is not installed. Install it with: pip install rembg"
                return [error]
            except (OSError, ValueError, RuntimeError) as exc:
                warnings.append({"path": image, "issue": "preprocess_failed", "detail": str(exc)})
        return [
            {
                "type": "success",
                "command": "preprocess_images",
                "processedImages": processed,
                "warnings": warnings,
                "stats": {"processed": len(processed), "failed": len(warnings)},
            }
        ]

    return [make_error(cmd or "unknown", "RECONSTRUCTION_FAILED")]
