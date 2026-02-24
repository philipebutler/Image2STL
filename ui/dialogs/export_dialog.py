"""
Export dialog - STL export with options.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    QDoubleSpinBox,
    QComboBox,
)


class ExportDialog(QDialog):
    """Modal dialog for STL export with scale options."""

    def __init__(self, project_name: str = "model", default_scale_mm: float = 150.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export STL")
        self.setMinimumWidth(440)
        self._project_name = project_name
        self._output_path: Optional[Path] = None
        self._build_ui(default_scale_mm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def output_path(self) -> Optional[Path]:
        return self._output_path

    @property
    def scale_mm(self) -> float:
        return self._scale_spin.value()

    @property
    def scale_axis(self) -> str:
        return self._axis_combo.currentText()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ui(self, default_scale_mm: float):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        # Output path row
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)

        default_filename = f"{self._project_name}.stl"
        self._path_edit = QLineEdit(default_filename)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output)
        path_layout.addWidget(self._path_edit, 1)
        path_layout.addWidget(browse_btn)
        form.addRow("Output file:", path_row)

        # Scale
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(1.0, 10000.0)
        self._scale_spin.setSingleStep(10.0)
        self._scale_spin.setDecimals(0)
        self._scale_spin.setValue(default_scale_mm)
        form.addRow("Scale (mm):", self._scale_spin)

        # Axis
        self._axis_combo = QComboBox()
        self._axis_combo.addItems(["longest", "width", "height", "depth"])
        form.addRow("Axis:", self._axis_combo)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export STL",
            self._path_edit.text(),
            "STL Files (*.stl);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)

    def _on_accept(self):
        path_text = self._path_edit.text().strip()
        if not path_text:
            return
        self._output_path = Path(path_text)
        if not self._output_path.suffix:
            self._output_path = self._output_path.with_suffix(".stl")
        self.accept()
