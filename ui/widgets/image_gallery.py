"""
Image gallery widget - displays image thumbnails in a scrollable grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QLabel,
    QPushButton,
    QSizePolicy,
    QFrame,
)

# Register optional image format support so QPixmap can decode HEIC/HEIF and AVIF files.
try:
    from image2stl.engine import _ensure_heif_support, _ensure_avif_support
    _ensure_heif_support()
    _ensure_avif_support()
except Exception:
    # Optional format plugins are not installed – thumbnails for HEIC/HEIF/AVIF
    # will fall back to the "(preview unavailable)" placeholder text.
    pass

THUMBNAIL_WIDTH = 260
THUMBNAIL_HEIGHT = 140
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".avif"}


class _ImageTile(QFrame):
    """A single image tile showing a thumbnail and filename."""

    remove_requested = Signal(str)  # emits file path

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._display_file_path = file_path
        # Use a stylesheet-managed border so we can restore it precisely when
        # the 'processed' indicator is toggled off (setFrameStyle() uses the
        # internal QPainter which stylesheet rules override unpredictably).
        self._default_style = "QFrame { border: 1px solid #555; border-radius: 2px; }"
        self.setStyleSheet(self._default_style)
        self._is_processed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("background: #2a2a2a;")
        self._load_thumbnail(self._display_file_path)
        layout.addWidget(self._thumb)

        # Bottom row: filename + processed tag + remove button
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        name_label = QLabel(Path(file_path).name)
        name_label.setMaximumWidth(THUMBNAIL_WIDTH - 80)
        name_label.setStyleSheet("color: #ccc; font-size: 11px;")
        name_label.setWordWrap(False)
        name_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        row.addWidget(name_label, 1)

        self._processed_badge = QLabel("✓ processed")
        self._processed_badge.setStyleSheet(
            "color: #3a3; font-size: 10px; font-weight: bold;"
        )
        self._processed_badge.setVisible(False)
        row.addWidget(self._processed_badge)

        remove_btn = QPushButton("Remove")
        remove_btn.setFixedWidth(70)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.file_path))
        row.addWidget(remove_btn)

        layout.addLayout(row)

    def set_processed(self, is_processed: bool):
        """Toggle the visual 'processed' indicator on this tile."""
        self._is_processed = is_processed
        self._processed_badge.setVisible(is_processed)
        if is_processed:
            self.setStyleSheet(
                "QFrame { border: 2px solid #3a3; border-radius: 4px; }"
            )
        else:
            self.setStyleSheet(self._default_style)

    def set_display_path(self, display_path: str | None):
        """Set the image path used for thumbnail display.

        Args:
            display_path: Optional alternate path (e.g., processed image). If
                not provided, falls back to the original source path.
        """
        resolved_path = display_path if display_path else self.file_path
        if resolved_path == self._display_file_path:
            return
        self._display_file_path = resolved_path
        self._load_thumbnail(self._display_file_path)

    def _load_thumbnail(self, file_path: str):
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self._thumb.clear()
            pixmap = pixmap.scaled(
                THUMBNAIL_WIDTH,
                THUMBNAIL_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._thumb.setPixmap(pixmap)
        else:
            self._thumb.setPixmap(QPixmap())
            self._thumb.setText("(preview\nunavailable)")
            self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)


class ImageGallery(QWidget):
    """Scrollable gallery of image thumbnails.

    Emits ``images_changed`` whenever the image list changes.
    """

    images_changed = Signal(list)  # list[str] of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_paths: List[str] = []
        self._tiles: List[_ImageTile] = []
        self._processed_preview_map: Dict[str, str] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._container_layout.setSpacing(8)
        self._container_layout.setContentsMargins(4, 4, 4, 4)

        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

        # Enable drop events on this widget
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_images(self, file_paths: List[str]) -> int:
        """Add images to the gallery (skips duplicates and unsupported types).

        Args:
            file_paths: List of absolute file paths.

        Returns:
            Number of images actually added.
        """
        added = 0
        for fp in file_paths:
            if fp in self._image_paths:
                continue
            if Path(fp).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            self._add_tile(fp)
            added += 1

        if added:
            self._apply_processed_preview_map()
            self.images_changed.emit(list(self._image_paths))
        return added

    def remove_image(self, file_path: str):
        """Remove a single image from the gallery."""
        if file_path not in self._image_paths:
            return

        self._processed_preview_map.pop(file_path, None)

        for tile in list(self._tiles):
            if tile.file_path == file_path:
                self._container_layout.removeWidget(tile)
                tile.deleteLater()
                self._tiles.remove(tile)
                break

        self._image_paths.remove(file_path)
        self.images_changed.emit(list(self._image_paths))

    def clear(self):
        """Remove all images from the gallery."""
        for tile in self._tiles:
            self._container_layout.removeWidget(tile)
            tile.deleteLater()
        self._tiles.clear()
        self._image_paths.clear()
        self._processed_preview_map.clear()
        self.images_changed.emit([])

    @property
    def image_paths(self) -> List[str]:
        return list(self._image_paths)

    @property
    def image_count(self) -> int:
        return len(self._image_paths)

    def mark_processed(
        self,
        processed_source_paths: list,
        processed_preview_map: Dict[str, str] | None = None,
    ):
        """Mark tiles whose source paths have been preprocessed.

        Args:
            processed_source_paths: List of original source file paths that now
                have corresponding processed versions available. Pass an empty
                list to clear all processed indicators.
            processed_preview_map: Optional mapping from source image path to
                processed image path to use for thumbnail preview display.
        """
        processed_set = set(processed_source_paths)
        self._processed_preview_map = {
            str(src): str(dst)
            for src, dst in (processed_preview_map or {}).items()
            if str(src) in self._image_paths
        }
        for tile in self._tiles:
            tile.set_processed(tile.file_path in processed_set)
            tile.set_display_path(self._processed_preview_map.get(tile.file_path))

    def _apply_processed_preview_map(self):
        """Apply current source→processed preview mapping to all tiles."""
        for tile in self._tiles:
            tile.set_display_path(self._processed_preview_map.get(tile.file_path))

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("border: 2px solid #0078d7;")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("")
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.add_images(paths)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_tile(self, file_path: str):
        tile = _ImageTile(file_path)
        tile.remove_requested.connect(self.remove_image)
        self._container_layout.addWidget(tile)
        self._tiles.append(tile)
        self._image_paths.append(file_path)
