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
    QSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QFrame,
)


class ControlPanel(QWidget):
    """Control bar containing reconstruction controls and action buttons.

    Layout (vertical):
      Row 1 – Mode | Scale | Action buttons
      Row 2 – Foreground Isolation group (compact + collapsible advanced panel)
    """

    generate_requested = Signal()
    export_requested = Signal()
    cancel_requested = Signal()
    preprocess_requested = Signal()

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API – reconstruction
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
        """Enable/disable controls during reconstruction or preprocessing."""
        self._generate_btn.setEnabled(not is_processing)
        self._cancel_btn.setVisible(is_processing)
        self._export_btn.setEnabled(not is_processing)
        self._preprocess_btn.setEnabled(not is_processing)

    def enable_export(self, enabled: bool = True):
        """Enable or disable the Export STL button."""
        self._export_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Public API – foreground isolation
    # ------------------------------------------------------------------

    @property
    def auto_isolate_enabled(self) -> bool:
        return self._auto_isolate_cb.isChecked()

    @property
    def preprocess_source(self) -> str:
        """Returns 'original' or 'processed'."""
        return "processed" if self._source_combo.currentIndex() == 1 else "original"

    @property
    def preprocess_strength(self) -> float:
        return self._strength_spin.value()

    @property
    def hole_fill_enabled(self) -> bool:
        return self._hole_fill_cb.isChecked()

    @property
    def island_removal_threshold(self) -> int:
        return self._island_spin.value()

    @property
    def crop_padding(self) -> int:
        return self._crop_padding_spin.value()

    @property
    def edge_feather_radius(self) -> int:
        return self._edge_feather_spin.value()

    @property
    def contrast_strength(self) -> float:
        return self._contrast_spin.value()

    def set_processed_count(self, count: int):
        """Update the 'N processed' label in the isolation group."""
        if count > 0:
            self._processed_status_label.setText(f"✓ {count} processed")
            self._processed_status_label.setStyleSheet("color: #3a3; font-size: 11px;")
        else:
            self._processed_status_label.setText("No processed images")
            self._processed_status_label.setStyleSheet("color: #888; font-size: 11px;")

    def load_isolation_settings(
        self,
        auto_isolate: bool,
        strength: float,
        source: str,
        hole_fill: bool,
        island_threshold: int,
        crop_padding: int,
        edge_feather_radius: int = 2,
        contrast_strength: float = 0.0,
    ):
        """Restore isolation settings from a saved project into the UI controls.

        Args:
            auto_isolate: Whether Auto Isolate should be checked.
            strength: Preprocessing strength value (0.0–1.0).
            source: Source selector value – "original" or "processed".
            hole_fill: Whether Fill holes should be checked.
            island_threshold: Island removal threshold value.
            crop_padding: Crop padding value in pixels.
            edge_feather_radius: Edge feather radius in pixels.
            contrast_strength: Contrast enhancement strength (0.0–1.0).
        """
        self._auto_isolate_cb.setChecked(auto_isolate)
        self._strength_spin.setValue(strength)
        self._source_combo.setCurrentIndex(1 if source == "processed" else 0)
        self._hole_fill_cb.setChecked(hole_fill)
        self._island_spin.setValue(island_threshold)
        self._crop_padding_spin.setValue(crop_padding)
        self._edge_feather_spin.setValue(edge_feather_radius)
        self._contrast_spin.setValue(contrast_strength)

    # ------------------------------------------------------------------
    # Internal UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # ---- Row 1: Mode | Scale | Buttons ----
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

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
        top_row.addWidget(mode_group)

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

        top_row.addWidget(scale_group)

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

        top_row.addLayout(btn_layout)
        top_row.addStretch(1)
        outer.addLayout(top_row)

        # ---- Row 2: Foreground Isolation ----
        outer.addWidget(self._build_isolation_row())

    def _build_isolation_row(self) -> QGroupBox:
        """Build the Foreground Isolation group box with compact + advanced rows."""
        iso_group = QGroupBox("Foreground Isolation")
        iso_outer = QVBoxLayout(iso_group)
        iso_outer.setSpacing(4)
        iso_outer.setContentsMargins(8, 4, 8, 4)

        # Compact row
        compact = QHBoxLayout()
        compact.setSpacing(10)

        self._auto_isolate_cb = QCheckBox("Auto Isolate")
        self._auto_isolate_cb.setToolTip(
            "Automatically remove backgrounds before reconstruction"
        )
        compact.addWidget(self._auto_isolate_cb)

        self._preprocess_btn = QPushButton("Run Preprocess")
        self._preprocess_btn.setToolTip("Manually run foreground isolation on all images now")
        self._preprocess_btn.clicked.connect(self.preprocess_requested)
        compact.addWidget(self._preprocess_btn)

        compact.addWidget(QLabel("Source:"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(["Original", "Processed"])
        self._source_combo.setFixedWidth(110)
        self._source_combo.setToolTip(
            "Choose which image set to send to reconstruction"
        )
        compact.addWidget(self._source_combo)

        compact.addWidget(QLabel("Strength:"))
        self._strength_spin = QDoubleSpinBox()
        self._strength_spin.setRange(0.0, 1.0)
        self._strength_spin.setSingleStep(0.1)
        self._strength_spin.setValue(0.5)
        self._strength_spin.setDecimals(1)
        self._strength_spin.setFixedWidth(64)
        self._strength_spin.setToolTip("Background removal strength (0.0 = light, 1.0 = aggressive)")
        compact.addWidget(self._strength_spin)

        self._processed_status_label = QLabel("No processed images")
        self._processed_status_label.setStyleSheet("color: #888; font-size: 11px;")
        compact.addWidget(self._processed_status_label)

        self._advanced_toggle_btn = QPushButton("Advanced ▸")
        self._advanced_toggle_btn.setFixedWidth(100)
        self._advanced_toggle_btn.setCheckable(True)
        self._advanced_toggle_btn.setChecked(False)
        self._advanced_toggle_btn.clicked.connect(self._toggle_advanced)
        compact.addWidget(self._advanced_toggle_btn)

        compact.addStretch(1)
        iso_outer.addLayout(compact)

        # Advanced row (hidden by default)
        self._advanced_panel = QFrame()
        self._advanced_panel.setVisible(False)
        adv_layout = QHBoxLayout(self._advanced_panel)
        adv_layout.setSpacing(10)
        adv_layout.setContentsMargins(0, 0, 0, 0)

        self._hole_fill_cb = QCheckBox("Fill holes")
        self._hole_fill_cb.setChecked(True)
        self._hole_fill_cb.setToolTip("Fill small holes in the foreground mask")
        adv_layout.addWidget(self._hole_fill_cb)

        adv_layout.addWidget(QLabel("Island threshold:"))
        self._island_spin = QSpinBox()
        self._island_spin.setRange(0, 10000)
        self._island_spin.setValue(500)
        self._island_spin.setFixedWidth(80)
        self._island_spin.setToolTip(
            "Minimum pixel area for a foreground region to be kept (0 = keep all)"
        )
        adv_layout.addWidget(self._island_spin)

        adv_layout.addWidget(QLabel("Crop padding (px):"))
        self._crop_padding_spin = QSpinBox()
        self._crop_padding_spin.setRange(0, 200)
        self._crop_padding_spin.setValue(10)
        self._crop_padding_spin.setFixedWidth(64)
        self._crop_padding_spin.setToolTip("Extra pixels to leave around the tight foreground crop")
        adv_layout.addWidget(self._crop_padding_spin)

        adv_layout.addWidget(QLabel("Edge feather (px):"))
        self._edge_feather_spin = QSpinBox()
        self._edge_feather_spin.setRange(0, 20)
        self._edge_feather_spin.setValue(2)
        self._edge_feather_spin.setFixedWidth(64)
        self._edge_feather_spin.setToolTip(
            "Radius for alpha edge feathering to smooth mask boundaries (0 = off)"
        )
        adv_layout.addWidget(self._edge_feather_spin)

        adv_layout.addWidget(QLabel("Contrast:"))
        self._contrast_spin = QDoubleSpinBox()
        self._contrast_spin.setRange(0.0, 1.0)
        self._contrast_spin.setSingleStep(0.1)
        self._contrast_spin.setValue(0.0)
        self._contrast_spin.setDecimals(1)
        self._contrast_spin.setFixedWidth(64)
        self._contrast_spin.setToolTip(
            "Foreground contrast/sharpness enhancement (0.0 = none, 1.0 = maximum)"
        )
        adv_layout.addWidget(self._contrast_spin)

        adv_layout.addStretch(1)
        iso_outer.addWidget(self._advanced_panel)

        return iso_group

    def _toggle_advanced(self, checked: bool):
        self._advanced_panel.setVisible(checked)
        self._advanced_toggle_btn.setText("Advanced ▾" if checked else "Advanced ▸")
