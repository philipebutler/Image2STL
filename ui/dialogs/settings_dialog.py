"""
Settings dialog - application preferences.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QLabel,
    QTabWidget,
    QWidget,
    QDoubleSpinBox,
    QCheckBox,
    QSpinBox,
)


class SettingsDialog(QDialog):
    """Modal dialog for editing application settings."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        self._config = config
        self._build_ui()
        self._load_values()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # --- General tab ---
        general = QWidget()
        gen_form = QFormLayout(general)

        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(1.0, 10000.0)
        self._scale_spin.setSingleStep(10.0)
        self._scale_spin.setDecimals(1)
        gen_form.addRow("Default scale (mm):", self._scale_spin)

        tabs.addTab(general, "General")

        # --- Cloud API tab ---
        cloud = QWidget()
        cloud_form = QFormLayout(cloud)

        self._api_key_edit = QLineEdit()
        self._api_key_edit.setPlaceholderText("Enter Meshy.ai API key")
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        cloud_form.addRow("Meshy API key:", self._api_key_edit)

        cloud_form.addRow(
            QLabel("Obtain a key at https://www.meshy.ai")
        )

        tabs.addTab(cloud, "Cloud API")

        # --- UI tab ---
        ui_tab = QWidget()
        ui_form = QFormLayout(ui_tab)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(800, 7680)
        self._width_spin.setSingleStep(80)
        ui_form.addRow("Window width:", self._width_spin)

        self._height_spin = QSpinBox()
        self._height_spin.setRange(600, 4320)
        self._height_spin.setSingleStep(60)
        ui_form.addRow("Window height:", self._height_spin)

        self._show_grid_cb = QCheckBox("Show grid")
        ui_form.addRow(self._show_grid_cb)

        self._show_axes_cb = QCheckBox("Show axes")
        ui_form.addRow(self._show_axes_cb)

        tabs.addTab(ui_tab, "UI")

        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_values(self):
        self._scale_spin.setValue(self._config.get("defaults.scale_mm", 150.0))
        self._api_key_edit.setText(self._config.get("meshy_api.api_key", ""))
        self._width_spin.setValue(self._config.get("ui.window_width", 1280))
        self._height_spin.setValue(self._config.get("ui.window_height", 800))
        self._show_grid_cb.setChecked(self._config.get("ui.show_grid", True))
        self._show_axes_cb.setChecked(self._config.get("ui.show_axes", True))

    def _on_accept(self):
        self._config.set("defaults.scale_mm", self._scale_spin.value())
        self._config.set("meshy_api.api_key", self._api_key_edit.text().strip())
        self._config.set("ui.window_width", self._width_spin.value())
        self._config.set("ui.window_height", self._height_spin.value())
        self._config.set("ui.show_grid", self._show_grid_cb.isChecked())
        self._config.set("ui.show_axes", self._show_axes_cb.isChecked())
        self._config.save()
        self.accept()
