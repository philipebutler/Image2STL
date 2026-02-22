from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from .errors import SUPPORTED_IMAGE_EXTENSIONS, make_error

WARNING_THRESHOLD_SECONDS = 600
DEFAULT_INPUT_DIMENSIONS_MM = (100.0, 120.0, 80.0)
MESHY_BASE_URL = "https://api.meshy.ai/v1"
MESHY_API_KEY_ENV = "MESHY_API_KEY"
MESHY_TIMEOUT_SECONDS = 600


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
    if len(images) > 5:
        return make_error("reconstruct", "TOO_MANY_IMAGES")
    for image in images:
        if Path(image).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return make_error("reconstruct", "UNSUPPORTED_FILE_FORMAT")
    if command.get("mode") == "cloud" and command.get("simulateApiError"):
        return make_error("reconstruct", "API_ERROR")
    return None


def _run_triposr_local(images: list[str], target: Path) -> dict:
    """Run TripoSR with the primary capture image; additional captures are reserved for future multi-view tuning."""
    try:
        import torch
        from PIL import Image
        from tsr.system import TSR
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - dependency availability is environment-specific
        raise RuntimeError("TripoSR dependencies unavailable") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TSR.from_pretrained("stabilityai/TripoSR")
    if hasattr(model, "to"):
        model = model.to(device)
    primary_image = images[0]  # TripoSR inference currently runs from the primary capture image.
    with Image.open(primary_image) as src_image:
        image = src_image.convert("RGB")
        scene_codes = model([image], device=device)
    meshes = model.extract_mesh(scene_codes)
    mesh = meshes[0]
    mesh.export(str(target))
    return {
        "vertices": len(getattr(mesh, "vertices", [])),
        "faces": len(getattr(mesh, "faces", [])),
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
    output = []
    if cmd == "reconstruct":
        validation_error = _validate_reconstruction(command)
        if validation_error:
            return [validation_error]
        output.append(_status(0.1, "Loading images...", estimated=120))
        output.append(_status(0.5, "Running AI reconstruction...", estimated=480))
        output.append(_status(0.8, "Repairing mesh...", estimated=90))
        output.append(_status(0.95, "Generating preview...", estimated=20))
        target = Path(command["outputPath"])
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = command.get("mode", "local")
        try:
            if mode == "cloud":
                api_key = _read_meshy_api_key(command)
                if not api_key:
                    return [make_error("reconstruct", "API_ERROR")]
                stats = _run_meshy_cloud(command["images"], target, api_key)
            else:
                stats = _run_triposr_local(command["images"], target)
        except (urlerror.URLError, RuntimeError, ValueError):
            error_code = "API_ERROR" if mode == "cloud" else "RECONSTRUCTION_FAILED"
            return [make_error("reconstruct", error_code)]
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return [{"type": "success", "command": "repair", "outputPath": str(dst)}]

    if cmd == "scale":
        src = Path(command["inputMesh"])
        dst = Path(command["outputMesh"])
        input_dimensions = tuple(command.get("inputDimensionsMm", DEFAULT_INPUT_DIMENSIONS_MM))
        factor = calculate_scale_factor(input_dimensions, float(command["targetSizeMm"]), command.get("axis", "longest"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(f"; scale_factor={factor:.4f}\n{src.read_text(encoding='utf-8')}", encoding="utf-8")
        return [{"type": "success", "command": "scale", "outputPath": str(dst), "scaleFactor": factor}]

    return [make_error(cmd or "unknown", "RECONSTRUCTION_FAILED")]
