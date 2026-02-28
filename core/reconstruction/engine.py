"""
ReconstructionEngine — orchestrates multi-method fallback and post-processing.

Attempts reconstruction methods in priority order (E → D → C → Cloud),
falling back automatically when a method cannot run or produces an error.
A successful result is then passed through a post-processing pipeline
(repair → optimize → scale → export).

This module is intentionally UI-framework agnostic: progress and events are
delivered through plain Python callbacks so the engine remains independently
testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod
from core.reconstruction.methods.method_e_hybrid import MethodEHybrid
from core.reconstruction.methods.method_d_dust3r import MethodDDust3R
from core.reconstruction.methods.method_c_triposr import MethodCTripoSR
from core.reconstruction.methods.method_cloud import MethodCloud

logger = logging.getLogger(__name__)

# Type alias for progress callbacks used throughout this module.
_ProgressCB = Optional[Callable[[int, str], None]]


@dataclass
class MethodAttempt:
    """Record of a single method attempt within a reconstruction run."""

    method: ReconstructionMethod
    result: ReconstructionResult


class ReconstructionEngine:
    """Orchestrates reconstruction with automatic method fallback.

    Attempts each method in the supplied (or auto-selected) chain in order.
    The first method that both *can run* and *succeeds* triggers the
    post-processing pipeline.  If every method fails the engine returns a
    consolidated failure result describing each error.

    Progress and lifecycle events are delivered through optional callbacks:

    * ``on_progress(percent: int, status: str)``
    * ``on_method_started(method_name: str)``
    * ``on_method_completed(method_name: str, success: bool)``
    """

    def __init__(self, config=None):
        """Initialise the engine.

        Args:
            config: Application config object (supports ``config.get(key, default)``).
                    May be ``None`` when not required.
        """
        self.config = config
        self.attempts: List[MethodAttempt] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        images: List[Path],
        output_dir: Path,
        method_chain: Optional[List[ReconstructionMethod]] = None,
        on_progress: _ProgressCB = None,
        on_method_started: Optional[Callable[[str], None]] = None,
        on_method_completed: Optional[Callable[[str, bool], None]] = None,
    ) -> ReconstructionResult:
        """Attempt reconstruction, falling back through the method chain.

        Args:
            images: Input image paths.
            output_dir: Directory for all output files.
            method_chain: Ordered list of :class:`ReconstructionMethod` values
                to try.  When ``None`` the chain is auto-selected from detected
                hardware capabilities.
            on_progress: Called with ``(percent, status)`` as work progresses.
            on_method_started: Called with the method name just before each
                attempt begins.
            on_method_completed: Called with ``(method_name, success)`` when an
                attempt finishes.

        Returns:
            :class:`ReconstructionResult` — either the first successful result
            (post-processed) or a consolidated failure result.
        """
        self.attempts = []

        if method_chain is None:
            hw = MethodSelector.detect_hardware()
            method_chain = MethodSelector.select_method(hw, num_images=len(images))

        logger.info("Method chain: %s", [m.value for m in method_chain])

        for method_enum in method_chain:
            reconstructor = self._get_reconstructor(method_enum)

            if on_progress:
                reconstructor.set_progress_callback(on_progress)

            can_run, reason = reconstructor.can_run()
            if not can_run:
                logger.warning("Skipping %s: %s", method_enum.value, reason)
                continue

            method_name = reconstructor.get_method_name()
            if on_method_started:
                on_method_started(method_name)

            result = reconstructor.reconstruct(images, output_dir)
            self.attempts.append(MethodAttempt(method=method_enum, result=result))

            if on_method_completed:
                on_method_completed(method_name, result.success)

            if result.success:
                logger.info(
                    "%s succeeded in %.1fs",
                    method_name,
                    result.processing_time_seconds,
                )
                final_path = self._post_process(result.mesh_path, output_dir, on_progress)
                return ReconstructionResult(
                    success=True,
                    mesh_path=final_path,
                    method_used=method_name,
                    processing_time_seconds=result.processing_time_seconds,
                    quality_score=result.quality_score,
                    metadata=result.metadata,
                )

            logger.warning("%s failed: %s", method_name, result.error_message)

        return self._build_all_failed_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reconstructor(self, method: ReconstructionMethod) -> BaseReconstructor:
        """Instantiate the reconstructor for *method*."""
        _registry = {
            ReconstructionMethod.METHOD_E: MethodEHybrid,
            ReconstructionMethod.METHOD_D: MethodDDust3R,
            ReconstructionMethod.METHOD_C: MethodCTripoSR,
            ReconstructionMethod.METHOD_CLOUD: MethodCloud,
        }
        return _registry[method](self.config)

    def _post_process(
        self,
        mesh_path: Path,
        output_dir: Path,
        on_progress: _ProgressCB = None,
    ) -> Path:
        """Post-processing pipeline: repair → optimize → scale → export.

        Each step is a stub that returns its input unchanged until Phase 5
        provides real implementations.  The pipeline order is fixed here so
        Phase 5 only needs to override the individual step methods.

        Args:
            mesh_path: Path to the raw reconstructed mesh.
            output_dir: Directory to write intermediate and final files.
            on_progress: Optional progress callback.

        Returns:
            Path to the final processed mesh file.
        """
        self._emit(on_progress, 90, "Post-processing: repairing mesh…")
        repaired = self._repair(mesh_path, output_dir / "repaired.obj")

        self._emit(on_progress, 93, "Post-processing: optimizing mesh…")
        optimized = self._optimize(repaired, output_dir / "optimized.obj")

        self._emit(on_progress, 96, "Post-processing: scaling and exporting…")
        scale_mm = (
            self.config.get("defaults.scale_mm", 150.0) if self.config else 150.0
        )
        final = self._scale_and_export(optimized, output_dir / "final.stl", scale_mm)

        self._emit(on_progress, 100, "Complete")
        return final

    # ------ individual post-processing steps (overridable in Phase 5) ------

    def _repair(self, src: Path, dst: Path) -> Path:
        """Repair mesh to make it watertight (stub — returns *src* until Phase 5)."""
        logger.debug("repair stub: %s → %s (no-op)", src, dst)
        return src

    def _optimize(self, src: Path, dst: Path) -> Path:
        """Simplify/optimise mesh (stub — returns *src* until Phase 5)."""
        logger.debug("optimize stub: %s → %s (no-op)", src, dst)
        return src

    def _scale_and_export(self, src: Path, dst: Path, scale_mm: float) -> Path:
        """Scale and export as STL (stub — returns *src* until Phase 5)."""
        logger.debug(
            "scale_and_export stub: %s → %s @ %.1f mm (no-op)", src, dst, scale_mm
        )
        return src

    # ------ utility ------

    def _emit(self, cb: _ProgressCB, percent: int, status: str) -> None:
        """Fire progress callback and log at INFO level."""
        if cb:
            cb(percent, status)
        logger.info("%d%%: %s", percent, status)

    def _build_all_failed_result(self) -> ReconstructionResult:
        """Build a consolidated failure :class:`ReconstructionResult`."""
        parts = []
        for attempt in self.attempts:
            reqs = MethodSelector.get_method_requirements(attempt.method)
            name = reqs.get("name", attempt.method.value)
            parts.append(f"{name}: {attempt.result.error_message}")

        message = (
            "All reconstruction methods failed:\n\n" + "\n".join(parts)
            if parts
            else "No reconstruction methods could run on this system"
        )
        logger.error(message)
        return ReconstructionResult(
            success=False,
            mesh_path=None,
            method_used="none",
            processing_time_seconds=0.0,
            error_message=message,
        )
