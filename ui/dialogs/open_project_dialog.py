"""
Open project dialog - lists existing projects for the user to select.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

PROJECT_FILE_NAME = "project.json"


class OpenProjectDialog(QDialog):
    """Modal dialog for opening an existing project."""

    def __init__(self, project_dirs: List[Path], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Project")
        self.setMinimumSize(480, 360)
        self._selected_path: Optional[Path] = None
        self._build_ui(project_dirs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selected_project_path(self) -> Optional[Path]:
        return self._selected_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ui(self, project_dirs: List[Path]):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a project to open:"))

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_double_click)

        for path in project_dirs:
            name = self._read_project_name(path)
            item = QListWidgetItem(f"{name}  ({path})")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._list.addItem(item)

        layout.addWidget(self._list)

        # Browse button
        browse_btn = QPushButton("Browse for project folder…")
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _read_project_name(self, path: Path) -> str:
        try:
            data = json.loads((path / PROJECT_FILE_NAME).read_text(encoding="utf-8"))
            return str(data.get("name", path.name))
        except Exception:
            return path.name

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(self, "Select project folder")
        if directory:
            self._selected_path = Path(directory)
            self.accept()

    def _on_double_click(self, item: QListWidgetItem):
        self._selected_path = Path(item.data(Qt.ItemDataRole.UserRole))
        self.accept()

    def _on_accept(self):
        current = self._list.currentItem()
        if current:
            self._selected_path = Path(current.data(Qt.ItemDataRole.UserRole))
            self.accept()
