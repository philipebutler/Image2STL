"""
Control panel widget - reconstruction mode, scale, and action buttons.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QGroupBox,
)


class ControlPanel(QWidget):
    """Horizontal bar containing reconstruction controls and action buttons."""

    generate_requested = Signal()
    export_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def reconstruction_mode(self) -> str:
        return "cloud" if self._cloud_radio.isChecked() else "local"

    @property
    def scale_mm(self) -> float:
        return self._scale_spin.value()

    @property
    def scale_axis(self) -> str:
        return self._axis_combo.currentText()

    def set_processing(self, is_processing: bool):
        """Enable/disable controls during reconstruction."""
        self._generate_btn.setEnabled(not is_processing)
        self._cancel_btn.setVisible(is_processing)
        self._export_btn.setEnabled(not is_processing)

    # ------------------------------------------------------------------
    # Internal UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)

        # --- Reconstruction mode ---
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.setSpacing(12)

        self._local_radio = QRadioButton("Local")
        self._local_radio.setChecked(True)
        self._cloud_radio = QRadioButton("Cloud")

        btn_group = QButtonGroup(self)
        btn_group.addButton(self._local_radio)
        btn_group.addButton(self._cloud_radio)

        mode_layout.addWidget(self._local_radio)
        mode_layout.addWidget(self._cloud_radio)
        main_layout.addWidget(mode_group)

        # --- Scale controls ---
        scale_group = QGroupBox("Scale")
        scale_layout = QHBoxLayout(scale_group)
        scale_layout.setSpacing(8)

        scale_layout.addWidget(QLabel("Size (mm):"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setMinimum(1.0)
        self._scale_spin.setMaximum(10000.0)
        self._scale_spin.setSingleStep(10.0)
        self._scale_spin.setValue(150.0)
        self._scale_spin.setDecimals(0)
        self._scale_spin.setFixedWidth(90)
        scale_layout.addWidget(self._scale_spin)

        scale_layout.addWidget(QLabel("Axis:"))
        self._axis_combo = QComboBox()
        self._axis_combo.addItems(["longest", "width", "height", "depth"])
        self._axis_combo.setFixedWidth(100)
        scale_layout.addWidget(self._axis_combo)

        main_layout.addWidget(scale_group)

        # --- Action buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self._generate_btn = QPushButton("Generate 3D Model")
        self._generate_btn.setDefault(True)
        self._generate_btn.clicked.connect(self.generate_requested)
        btn_layout.addWidget(self._generate_btn)

        self._export_btn = QPushButton("Export STL")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self.export_requested)
        btn_layout.addWidget(self._export_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self.cancel_requested)
        btn_layout.addWidget(self._cancel_btn)

        main_layout.addLayout(btn_layout)
        main_layout.addStretch(1)

    def enable_export(self, enabled: bool = True):
        """Enable or disable the Export STL button."""
        self._export_btn.setEnabled(enabled)
