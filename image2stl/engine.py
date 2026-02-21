from __future__ import annotations

import json
from pathlib import Path

from .errors import SUPPORTED_IMAGE_EXTENSIONS, make_error


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
        if estimated > 600:
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
            return make_error("reconstruct", "INSUFFICIENT_FEATURES")
    if command.get("mode") == "cloud" and command.get("simulateApiError"):
        return make_error("reconstruct", "API_ERROR")
    return None


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
        target.write_text("o mock_reconstruction\nv 0 0 0\n", encoding="utf-8")
        output.append(
            {
                "type": "success",
                "command": "reconstruct",
                "outputPath": str(target),
                "stats": {"vertices": 1, "faces": 0},
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
        factor = calculate_scale_factor((100.0, 120.0, 80.0), float(command["targetSizeMm"]), command.get("axis", "longest"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(f"; scale_factor={factor:.4f}\n{src.read_text(encoding='utf-8')}", encoding="utf-8")
        return [{"type": "success", "command": "scale", "outputPath": str(dst), "scaleFactor": factor}]

    return [make_error(cmd or "unknown", "RECONSTRUCTION_FAILED")]
