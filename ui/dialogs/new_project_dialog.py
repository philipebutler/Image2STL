"""
New project dialog - creates a new project with a name and directory.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class NewProjectDialog(QDialog):
    """Modal dialog for creating a new project."""

    def __init__(self, default_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(420)
        self._default_dir = default_dir
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def project_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def project_dir(self) -> Path:
        return Path(self._dir_edit.text().strip())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. MyHeadScan")
        form.addRow("Project name:", self._name_edit)

        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self._dir_edit = QLineEdit(str(self._default_dir))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self._dir_edit, 1)
        dir_layout.addWidget(browse_btn)
        form.addRow("Location:", dir_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select project location", str(self._default_dir)
        )
        if directory:
            self._dir_edit.setText(directory)

    def _on_accept(self):
        name = self._name_edit.text().strip()
        if not name:
            self._name_edit.setPlaceholderText("Name is required")
            return
        # Reject names with path separators
        if "/" in name or "\\" in name:
            self._name_edit.setPlaceholderText("Name must not contain path separators")
            self._name_edit.clear()
            return
        self.accept()
