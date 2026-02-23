"""
Reconstruction engine - integrates with image2stl backend
"""
from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from typing import Callable, List, Optional
import logging

from image2stl.engine import process_command

logger = logging.getLogger(__name__)


class ReconstructionEngine:
    """Wrapper around the image2stl engine for use with the PySide6 UI.

    Runs reconstruction in a background thread so the UI remains responsive.
    Progress and completion are reported via callbacks rather than Qt signals
    to keep this class independent of PySide6.
    """

    def __init__(self, config=None):
        self._config = config
        self._current_operation_id: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def reconstruct(
        self,
        images: List[str],
        output_path: Path,
        mode: str = "local",
        api_key: Optional[str] = None,
        on_progress: Optional[Callable[[float, str, Optional[int]], None]] = None,
        on_success: Optional[Callable[[str, dict], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ):
        """Start reconstruction in a background thread.

        Args:
            images: List of absolute image paths.
            output_path: Destination path for the output mesh.
            mode: "local" or "cloud".
            api_key: Meshy API key (cloud mode only).
            on_progress: Called with (progress_fraction, status_text, estimated_seconds).
            on_success: Called with (output_path, stats_dict).
            on_error: Called with (error_code, message).
        """
        if self.is_running:
            logger.warning("Reconstruction already in progress")
            return

        self._current_operation_id = str(uuid.uuid4())

        command: dict = {
            "command": "reconstruct",
            "mode": mode,
            "images": images,
            "outputPath": str(output_path),
            "operationId": self._current_operation_id,
        }
        if api_key:
            command["apiKey"] = api_key

        def _run():
            messages = process_command(command)
            for msg in messages:
                msg_type = msg.get("type")
                if msg_type == "progress":
                    if on_progress:
                        on_progress(
                            msg.get("progress", 0.0),
                            msg.get("status", ""),
                            msg.get("estimatedSecondsRemaining"),
                        )
                elif msg_type == "success":
                    if on_success:
                        on_success(msg.get("outputPath", ""), msg.get("stats", {}))
                elif msg_type == "error":
                    if on_error:
                        on_error(msg.get("errorCode", "UNKNOWN_ERROR"), msg.get("message", ""))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self):
        """Cancel the current reconstruction operation."""
        if self._current_operation_id:
            process_command({
                "command": "cancel",
                "operationId": self._current_operation_id,
            })
            self._current_operation_id = None

    def repair(
        self,
        input_mesh: Path,
        output_mesh: Path,
        target_face_count: int = 100000,
        on_success: Optional[Callable[[str, dict], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ):
        """Repair a mesh in a background thread.

        Args:
            input_mesh: Path to input mesh file.
            output_mesh: Path for repaired output.
            target_face_count: Maximum faces in output mesh.
            on_success: Called with (output_path, stats_dict).
            on_error: Called with (error_code, message).
        """
        command = {
            "command": "repair",
            "inputMesh": str(input_mesh),
            "outputMesh": str(output_mesh),
            "targetFaceCount": target_face_count,
        }

        def _run():
            messages = process_command(command)
            for msg in messages:
                if msg.get("type") == "success":
                    if on_success:
                        on_success(msg.get("outputPath", ""), msg.get("stats", {}))
                elif msg.get("type") == "error":
                    if on_error:
                        on_error(msg.get("errorCode", "UNKNOWN_ERROR"), msg.get("message", ""))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def scale(
        self,
        input_mesh: Path,
        output_mesh: Path,
        target_size_mm: float,
        axis: str = "longest",
        input_dimensions_mm: tuple = (100.0, 120.0, 80.0),
        on_success: Optional[Callable[[str, float], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ):
        """Scale a mesh in a background thread.

        Args:
            input_mesh: Path to input mesh file.
            output_mesh: Path for scaled output.
            target_size_mm: Target size in millimetres.
            axis: Axis to scale to ("longest", "width", "height", "depth").
            input_dimensions_mm: Current (width, height, depth) in mm.
            on_success: Called with (output_path, scale_factor).
            on_error: Called with (error_code, message).
        """
        command = {
            "command": "scale",
            "inputMesh": str(input_mesh),
            "outputMesh": str(output_mesh),
            "targetSizeMm": target_size_mm,
            "axis": axis,
            "inputDimensionsMm": list(input_dimensions_mm),
        }

        def _run():
            messages = process_command(command)
            for msg in messages:
                if msg.get("type") == "success":
                    if on_success:
                        on_success(msg.get("outputPath", ""), msg.get("scaleFactor", 1.0))
                elif msg.get("type") == "error":
                    if on_error:
                        on_error(msg.get("errorCode", "UNKNOWN_ERROR"), msg.get("message", ""))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
