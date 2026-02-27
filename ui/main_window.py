"""
Main application window
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot, QMetaObject, Q_ARG
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QLabel,
    QPushButton,
)
import logging

from config import Config
from core.project_manager import ProjectManager
from core.reconstruction_engine import ReconstructionEngine
from ui.widgets.image_gallery import ImageGallery
from ui.widgets.viewer_3d import Viewer3D
from ui.widgets.control_panel import ControlPanel
from ui.widgets.progress_widget import ProgressWidget
from ui.dialogs.new_project_dialog import NewProjectDialog
from ui.dialogs.open_project_dialog import OpenProjectDialog
from ui.dialogs.settings_dialog import SettingsDialog
from ui.dialogs.export_dialog import ExportDialog

logger = logging.getLogger(__name__)

_SETTINGS_HASH_SUFFIX_RE = re.compile(r"_[0-9a-fA-F]{8}$")


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.project_manager = ProjectManager(Path(config.get("app.last_project_dir")))
        self.reconstruction_engine = ReconstructionEngine(config)
        self._reconstructed_model_path: Optional[Path] = None
        # Tracks the source paths whose processed versions are available for
        # the current session (populated by _on_preprocess_success).
        self._processed_source_paths: list[str] = []
        # Tracks the actual processed file paths returned by the engine.
        self._processed_image_paths: list[str] = []
        # When True, a preprocess run triggered automatically should chain
        # directly into reconstruction once it completes.
        self._preprocess_then_reconstruct: bool = False

        self._init_ui()
        self._connect_signals()
        self._restore_window_state()

    # ------------------------------------------------------------------
    # UI initialisation
    # ------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle("Image2STL")
        self.setMinimumSize(1024, 768)

        self._create_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Horizontal splitter: left panel | 3D viewer
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # 3D viewer
        self.viewer_3d = Viewer3D(self.config)
        splitter.addWidget(self.viewer_3d)
        splitter.setSizes([320, 700])

        main_layout.addWidget(splitter, 1)

        # Control panel (mode, scale, isolation, buttons)
        self.control_panel = ControlPanel(self.config)
        main_layout.addWidget(self.control_panel)

        # Progress + error display
        self.progress_widget = ProgressWidget()
        self.progress_widget.setVisible(False)
        main_layout.addWidget(self.progress_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Drag and drop 3-50 images or use File → Add Images.")

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Images (3–50)"))

        self.image_gallery = ImageGallery()
        layout.addWidget(self.image_gallery, 1)

        add_btn = QPushButton("Add Images…")
        add_btn.clicked.connect(self._on_add_images)
        layout.addWidget(add_btn)

        return panel

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        new_action = QAction("&New Project…", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        settings_action = QAction("Se&ttings…", self)
        settings_action.triggered.connect(self._on_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Images menu
        images_menu = menu_bar.addMenu("&Images")
        add_images_action = QAction("&Add Images…", self)
        add_images_action.setShortcut("Ctrl+I")
        add_images_action.triggered.connect(self._on_add_images)
        images_menu.addAction(add_images_action)

        # Preprocess menu entry
        preprocess_action = QAction("&Isolate Foreground", self)
        preprocess_action.setShortcut("Ctrl+P")
        preprocess_action.triggered.connect(self._on_preprocess)
        images_menu.addAction(preprocess_action)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.image_gallery.images_changed.connect(self._on_images_changed)
        self.control_panel.generate_requested.connect(self._on_generate)
        self.control_panel.export_requested.connect(self._on_export)
        self.control_panel.cancel_requested.connect(self._on_cancel)
        self.control_panel.preprocess_requested.connect(self._on_preprocess)

    # ------------------------------------------------------------------
    # Window state
    # ------------------------------------------------------------------

    def _restore_window_state(self):
        w = self.config.get("ui.window_width", 1280)
        h = self.config.get("ui.window_height", 800)
        self.resize(w, h)

    # ------------------------------------------------------------------
    # Menu / toolbar handlers
    # ------------------------------------------------------------------

    @Slot()
    def _on_new_project(self):
        dialog = NewProjectDialog(
            default_dir=Path(self.config.get("app.last_project_dir")),
            parent=self,
        )
        if dialog.exec() == NewProjectDialog.DialogCode.Accepted:
            try:
                project = self.project_manager.create_project(
                    dialog.project_name, dialog.project_dir
                )
                self.image_gallery.clear()
                self.progress_widget.reset()
                self.progress_widget.setVisible(False)
                self._reconstructed_model_path = None
                self._processed_source_paths = []
                self._processed_image_paths = []
                self.viewer_3d.reset_placeholder()
                self.control_panel.enable_export(False)
                self.control_panel.set_processed_count(0)
                self.setWindowTitle(f"Image2STL – {project.name}")
                self.status_bar.showMessage(f"New project '{project.name}' created.")
                self.config.set("app.last_project_dir", str(dialog.project_dir))
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Could not create project:\n{exc}")

    @Slot()
    def _on_open_project(self):
        projects = self.project_manager.list_projects()
        dialog = OpenProjectDialog(projects, parent=self)
        if dialog.exec() == OpenProjectDialog.DialogCode.Accepted and dialog.selected_project_path:
            self._load_project(dialog.selected_project_path)

    def _load_project(self, project_path: Path):
        try:
            project = self.project_manager.open_project(project_path)
            self.image_gallery.clear()
            image_paths = [str(p) for p in project.get_image_paths() if p.exists()]
            self.image_gallery.add_images(image_paths)
            model_path = project.get_model_path()
            self._reconstructed_model_path = model_path if model_path and model_path.exists() else None
            if self._reconstructed_model_path:
                self.viewer_3d.load_model(self._reconstructed_model_path)
                self.control_panel.enable_export(True)
            else:
                self.viewer_3d.reset_placeholder()
                self.control_panel.enable_export(False)

            # Restore processed images state
            processed_abs = [
                str((project.project_path / p).resolve())
                for p in project.processed_images
                if (project.project_path / p).exists()
            ]
            self._processed_image_paths = processed_abs

            preview_map = self._build_processed_preview_map(image_paths, processed_abs)
            self._processed_source_paths = list(preview_map.keys())
            self.image_gallery.mark_processed(
                self._processed_source_paths,
                processed_preview_map=preview_map,
            )
            self.control_panel.set_processed_count(len(processed_abs))

            # Restore isolation settings from project
            s = project.settings
            self.control_panel.load_isolation_settings(
                auto_isolate=s.auto_isolate_enabled,
                strength=s.preprocess_strength,
                source=s.preprocess_source_mode,
                hole_fill=s.hole_fill_enabled,
                island_threshold=s.island_removal_threshold,
                crop_padding=s.crop_padding,
                edge_feather_radius=s.edge_feather_radius,
                contrast_strength=s.contrast_strength,
            )

            self.setWindowTitle(f"Image2STL – {project.name}")
            self.status_bar.showMessage(f"Loaded project '{project.name}'.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not open project:\n{exc}")

    @Slot()
    def _on_save_project(self):
        if self.project_manager.current_project:
            try:
                self._sync_preprocess_settings_to_project()
                self.project_manager.save_current_project()
                self.status_bar.showMessage("Project saved.")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Could not save project:\n{exc}")
        else:
            self.status_bar.showMessage("No active project to save.")

    def _sync_preprocess_settings_to_project(self):
        """Write control panel preprocessing settings into the current project."""
        project = self.project_manager.current_project
        if project is None:
            return
        s = project.settings
        s.auto_isolate_enabled = self.control_panel.auto_isolate_enabled
        s.preprocess_strength = self.control_panel.preprocess_strength
        s.preprocess_source_mode = self.control_panel.preprocess_source
        s.hole_fill_enabled = self.control_panel.hole_fill_enabled
        s.island_removal_threshold = self.control_panel.island_removal_threshold
        s.crop_padding = self.control_panel.crop_padding
        s.edge_feather_radius = self.control_panel.edge_feather_radius
        s.contrast_strength = self.control_panel.contrast_strength

    @Slot()
    def _on_settings(self):
        dialog = SettingsDialog(self.config, parent=self)
        dialog.exec()

    @Slot()
    def _on_add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select source images",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.heic *.heif *.webp *.avif);;All Files (*)",
        )
        if paths:
            added = self.image_gallery.add_images(paths)
            self.status_bar.showMessage(f"Added {added} image(s).")

    # ------------------------------------------------------------------
    # Image gallery changes
    # ------------------------------------------------------------------

    @Slot(list)
    def _on_images_changed(self, image_paths: list):
        count = len(image_paths)
        if count < 3:
            msg = f"Loaded {count} image(s). Add at least {3 - count} more."
        elif count > 5:
            msg = f"Loaded {count} image(s). Use 3–50 images for best results."
        else:
            msg = f"Loaded {count} image(s). Ready for reconstruction."
        self.status_bar.showMessage(msg)

        # Clear processed state when image list changes
        if self._processed_source_paths or self._processed_image_paths:
            self._processed_source_paths = []
            self._processed_image_paths = []
            self.image_gallery.mark_processed([], processed_preview_map={})
            self.control_panel.set_processed_count(0)

        # Keep project in sync
        if self.project_manager.current_project is not None:
            project = self.project_manager.current_project
            project.images = []
            for fp in image_paths:
                try:
                    rel = str(Path(fp).relative_to(project.project_path))
                    project.images.append(rel)
                except ValueError:
                    project.images.append(fp)

    # ------------------------------------------------------------------
    # Foreground isolation / preprocessing
    # ------------------------------------------------------------------

    @Slot()
    def _on_preprocess(self):
        """Manually triggered foreground isolation."""
        images = self.image_gallery.image_paths
        if not images:
            QMessageBox.warning(self, "No images", "Add at least one image before running preprocessing.")
            return

        self._preprocess_then_reconstruct = False
        self._start_preprocess(images)

    def _start_preprocess(self, images: list[str]):
        """Start preprocessing for the given image list."""
        project = self.project_manager.current_project
        if project:
            output_dir = project.project_path / "preview" / "processed"
        else:
            output_dir = Path.home() / "image2stl_processed"

        self.progress_widget.reset()
        self.progress_widget.setVisible(True)
        self.control_panel.set_processing(True)
        self.status_bar.showMessage("Isolating foreground…")

        self.reconstruction_engine.preprocess_images(
            images=images,
            output_dir=output_dir,
            strength=self.control_panel.preprocess_strength,
            hole_fill=self.control_panel.hole_fill_enabled,
            island_removal_threshold=self.control_panel.island_removal_threshold,
            crop_padding=self.control_panel.crop_padding,
            edge_feather_radius=self.control_panel.edge_feather_radius,
            contrast_strength=self.control_panel.contrast_strength,
            on_success=self._on_preprocess_success,
            on_error=self._on_preprocess_error,
        )

    def _on_preprocess_success(self, processed_paths: list, stats: dict):
        payload = json.dumps(
            {
                "processed_paths": [str(p) for p in (processed_paths or [])],
                "stats": stats or {},
            }
        )
        QMetaObject.invokeMethod(
            self,
            "_handle_preprocess_success",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, payload),
        )

    @Slot(str)
    def _handle_preprocess_success(self, payload: str):
        try:
            parsed = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            parsed = {}
        processed_paths = [str(p) for p in parsed.get("processed_paths", [])]
        stats = parsed.get("stats", {})
        failed = stats.get("failed", 0)
        images = self.image_gallery.image_paths
        self._processed_image_paths = processed_paths

        preview_map = self._build_processed_preview_map(images, processed_paths)
        if len(preview_map) != len(processed_paths):
            logger.warning(
                "Mapped %d processed preview(s) from %d processed image(s).",
                len(preview_map),
                len(processed_paths),
            )

        self._processed_source_paths = list(preview_map.keys())
        self.image_gallery.mark_processed(
            self._processed_source_paths,
            processed_preview_map=preview_map,
        )
        self.control_panel.set_processed_count(len(processed_paths))
        self.control_panel.set_processing(False)

        if failed > 0:
            self.progress_widget.show_error(
                f"{failed} image(s) could not be preprocessed.",
                "Check the log for details. Remaining images were processed successfully.",
            )
        else:
            self.progress_widget.set_complete()

        # Persist to project
        project = self.project_manager.current_project
        if project:
            project.processed_images = [
                str(Path(p).relative_to(project.project_path))
                for p in processed_paths
                if Path(p).is_relative_to(project.project_path)
            ]
            self._sync_preprocess_settings_to_project()
            try:
                self.project_manager.save_current_project()
            except Exception:
                pass

        n = len(processed_paths)
        self.status_bar.showMessage(
            f"Foreground isolation complete. {n} image(s) processed."
        )

        if self._preprocess_then_reconstruct:
            self._preprocess_then_reconstruct = False
            self._start_reconstruction(self._processed_image_paths)

    def _build_processed_preview_map(
        self,
        source_images: list[str],
        processed_paths: list[str],
    ) -> dict[str, str]:
        """Build a mapping from source image path to processed preview path."""
        source_map = {Path(src).stem: src for src in source_images}
        preview_map: dict[str, str] = {}
        unmatched_processed: list[str] = []

        for processed in processed_paths:
            processed_stem = Path(processed).stem
            source_stem = self._extract_source_stem_from_processed(processed_stem)
            source_path = source_map.get(source_stem)
            if source_path and source_path not in preview_map:
                preview_map[source_path] = processed
            else:
                unmatched_processed.append(processed)

        if unmatched_processed:
            remaining_sources = [src for src in source_images if src not in preview_map]
            for src, processed in zip(remaining_sources, unmatched_processed):
                preview_map[src] = processed

        return preview_map

    def _extract_source_stem_from_processed(self, processed_stem: str) -> str:
        """Extract the original source stem from a processed output stem."""
        base = processed_stem
        if base.endswith("_processed"):
            base = base[: -len("_processed")]
        return _SETTINGS_HASH_SUFFIX_RE.sub("", base)

    def _on_preprocess_error(self, error_code: str, message: str):
        payload = json.dumps({"error_code": error_code, "message": message})
        QMetaObject.invokeMethod(
            self,
            "_handle_preprocess_error",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, payload),
        )

    @Slot(str)
    def _handle_preprocess_error(self, payload: str):
        try:
            parsed = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            parsed = {}
        error_code = str(parsed.get("error_code", ""))
        message = str(parsed.get("message", "Unknown preprocessing error"))
        self.control_panel.set_processing(False)
        self._preprocess_then_reconstruct = False
        suggestions = {
            "REMBG_UNAVAILABLE": "Install rembg with: pip install rembg",
        }
        suggestion = suggestions.get(error_code, "Check the log for details.")
        self.progress_widget.show_error(message, suggestion)
        self.status_bar.showMessage(f"Preprocessing error: {message}")

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _resolve_reconstruction_images(self) -> list[str]:
        """Return the correct image list based on source selector and availability."""
        source = self.control_panel.preprocess_source
        if source == "processed" and self._processed_image_paths:
            return self._processed_image_paths
        return self.image_gallery.image_paths

    @Slot()
    def _on_generate(self):
        images = self.image_gallery.image_paths
        if len(images) < 3:
            QMessageBox.warning(self, "Not enough images", "Add at least 3 images before generating.")
            return
        if len(images) > 50:
            QMessageBox.warning(self, "Too many images", "Use 3–50 images for best results.")
            return

        # Auto-isolate: run preprocess first, then chain into reconstruction
        if self.control_panel.auto_isolate_enabled:
            self._preprocess_then_reconstruct = True
            self._start_preprocess(images)
            return

        # Use resolved source set (original or already-processed)
        source_images = self._resolve_reconstruction_images()
        self._start_reconstruction(source_images)

    def _start_reconstruction(self, images: list[str]):
        """Start the reconstruction pipeline with the given image list."""
        # Determine output path
        project = self.project_manager.current_project
        if project:
            output_path = project.models_dir / "raw_reconstruction.obj"
        else:
            output_path = Path.home() / "raw_reconstruction.obj"

        mode = self.control_panel.reconstruction_mode
        api_key = self.config.get("meshy_api.api_key") if mode == "cloud" else None

        self.progress_widget.reset()
        self.progress_widget.setVisible(True)
        self.control_panel.set_processing(True)
        self.status_bar.showMessage(f"Generating 3D model ({mode} mode)…")

        self.reconstruction_engine.reconstruct(
            images=images,
            output_path=output_path,
            mode=mode,
            api_key=api_key,
            on_progress=self._on_engine_progress,
            on_success=self._on_engine_success,
            on_error=self._on_engine_error,
        )

    def _on_engine_progress(self, fraction: float, status: str, estimated_seconds):
        # Called from background thread – use invokeMethod for thread safety
        estimated_seconds_int = int(estimated_seconds) if isinstance(estimated_seconds, (int, float)) else -1
        payload = json.dumps(
            {
                "fraction": float(fraction),
                "status": str(status),
                "estimated_seconds": estimated_seconds_int,
            }
        )
        QMetaObject.invokeMethod(
            self,
            "_update_progress",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, payload),
        )

    @Slot(str)
    def _update_progress(self, payload: str):
        try:
            parsed = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            parsed = {}
        fraction = float(parsed.get("fraction", 0.0))
        status = str(parsed.get("status", ""))
        estimated_seconds = int(parsed.get("estimated_seconds", -1))
        self.progress_widget.set_progress(fraction, status, None if estimated_seconds < 0 else estimated_seconds)

    def _on_engine_success(self, output_path_str: str, stats: dict):
        payload = json.dumps({"output_path": output_path_str})
        QMetaObject.invokeMethod(
            self,
            "_handle_success",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, payload),
        )

    @Slot(str)
    def _handle_success(self, payload: str):
        try:
            parsed = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            parsed = {}
        output_path_str = str(parsed.get("output_path", ""))
        self._reconstructed_model_path = Path(output_path_str) if output_path_str else None
        self.progress_widget.set_complete()
        self.control_panel.set_processing(False)
        loaded = False
        if self._reconstructed_model_path and self._reconstructed_model_path.exists():
            loaded = self.viewer_3d.load_model(self._reconstructed_model_path)
        if not loaded:
            self.viewer_3d.reset_placeholder()
        self.control_panel.enable_export(self._reconstructed_model_path is not None)
        self.status_bar.showMessage("Reconstruction complete. Ready for export.")

        if self.project_manager.current_project and self._reconstructed_model_path:
            try:
                self.project_manager.current_project.set_model(self._reconstructed_model_path)
                self._sync_preprocess_settings_to_project()
                self.project_manager.save_current_project()
            except Exception:
                pass

    def _on_engine_error(self, error_code: str, message: str):
        payload = json.dumps({"error_code": error_code, "message": message})
        QMetaObject.invokeMethod(
            self,
            "_handle_error",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, payload),
        )

    @Slot(str)
    def _handle_error(self, payload: str):
        try:
            parsed = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            parsed = {}
        error_code = str(parsed.get("error_code", ""))
        message = str(parsed.get("message", "Unknown reconstruction error"))
        self.control_panel.set_processing(False)
        suggestions = {
            "INSUFFICIENT_IMAGES": "Add at least 3 images.",
            "TOO_MANY_IMAGES": "Use 3–50 images.",
            "API_ERROR": "Check your API key in Settings or try Local mode.",
            "RECONSTRUCTION_FAILED": "Try different images or switch to Cloud mode.",
            "OPERATION_CANCELLED": "Operation was cancelled.",
        }
        suggestion = suggestions.get(error_code, "Check the log for details.")
        self.progress_widget.show_error(message, suggestion)
        self.status_bar.showMessage(f"Error: {message}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @Slot()
    def _on_export(self):
        project = self.project_manager.current_project
        project_name = project.name if project else "model"
        default_scale = self.control_panel.scale_mm

        dialog = ExportDialog(project_name=project_name, default_scale_mm=default_scale, parent=self)
        if dialog.exec() != ExportDialog.DialogCode.Accepted or not dialog.output_path:
            return

        if not self._reconstructed_model_path or not self._reconstructed_model_path.exists():
            QMessageBox.warning(
                self,
                "No model",
                "No reconstructed model is available. Generate a 3D model first.",
            )
            return

        output_path = dialog.output_path
        scale_mm = dialog.scale_mm
        scale_axis = dialog.scale_axis

        # Apply scaling via the engine and write directly to the chosen output path.
        self.status_bar.showMessage(f"Scaling and exporting to {output_path.name}…")
        self.control_panel.set_processing(True)

        self.reconstruction_engine.scale(
            input_mesh=self._reconstructed_model_path,
            output_mesh=output_path,
            target_size_mm=scale_mm,
            axis=scale_axis,
            on_success=self._on_export_success,
            on_error=self._on_export_error,
        )

    def _on_export_success(self, output_path_str: str, scale_factor: float):
        payload = json.dumps({"output_path": output_path_str, "scale_factor": float(scale_factor)})
        QMetaObject.invokeMethod(
            self,
            "_handle_export_success",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, payload),
        )

    @Slot(str)
    def _handle_export_success(self, payload: str):
        try:
            parsed = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            parsed = {}
        output_path_str = str(parsed.get("output_path", ""))
        scale_factor = float(parsed.get("scale_factor", 1.0))
        self.control_panel.set_processing(False)
        name = Path(output_path_str).name if output_path_str else "file"
        self.status_bar.showMessage(
            f"Exported to {name} (scale factor {scale_factor:.4f})"
        )

    def _on_export_error(self, error_code: str, message: str):
        payload = json.dumps({"error_code": error_code, "message": message})
        QMetaObject.invokeMethod(
            self,
            "_handle_export_error",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, payload),
        )

    @Slot(str)
    def _handle_export_error(self, payload: str):
        try:
            parsed = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            parsed = {}
        message = str(parsed.get("message", "Unknown export error"))
        self.control_panel.set_processing(False)
        QMessageBox.critical(self, "Export failed", message)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    @Slot()
    def _on_cancel(self):
        self.reconstruction_engine.cancel()
        self._preprocess_then_reconstruct = False
        self.control_panel.set_processing(False)
        self.progress_widget.setVisible(False)
        self.status_bar.showMessage("Operation cancelled.")

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.reconstruction_engine.is_running:
            self.reconstruction_engine.cancel()
            # Wait briefly for the daemon thread to flush any partial output.
            # The thread is a daemon so the interpreter will not hang if the
            # timeout expires.
            self.reconstruction_engine.wait_for_completion(timeout=2.0)
        self._sync_preprocess_settings_to_project()
        self.project_manager.close_current_project()
        self.config.set("ui.window_width", self.width())
        self.config.set("ui.window_height", self.height())
        self.config.save()
        super().closeEvent(event)
