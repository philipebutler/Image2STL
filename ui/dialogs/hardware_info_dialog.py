"""
Hardware info dialog - shows detected hardware capabilities for reconstruction.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from core.reconstruction.method_selector import (
    HardwareCapabilities,
    MethodSelector,
    ReconstructionMethod,
)


def _yes_no(value: bool) -> str:
    return "✓  Yes" if value else "✗  No"


class HardwareInfoDialog(QDialog):
    """Read-only dialog showing detected hardware capabilities.

    Args:
        capabilities: The :class:`HardwareCapabilities` instance to display.
        parent: Optional parent widget.
    """

    def __init__(self, capabilities: HardwareCapabilities, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware Capabilities")
        self.setMinimumWidth(420)
        self._capabilities = capabilities
        self._build_ui()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # GPU / Compute section
        gpu_group = QGroupBox("GPU / Compute")
        gpu_form = QFormLayout(gpu_group)
        gpu_form.addRow("Platform:", QLabel(self._capabilities.platform or "Unknown"))
        gpu_form.addRow("CUDA available:", QLabel(_yes_no(self._capabilities.has_cuda)))
        gpu_form.addRow(
            "Metal (MPS) available:", QLabel(_yes_no(self._capabilities.has_mps))
        )
        vram_text = (
            f"{self._capabilities.total_vram_gb:.1f} GB"
            if self._capabilities.total_vram_gb > 0
            else "—"
        )
        gpu_form.addRow("Total VRAM:", QLabel(vram_text))
        devices_text = (
            ", ".join(self._capabilities.cuda_devices)
            if self._capabilities.cuda_devices
            else "—"
        )
        gpu_form.addRow("Devices:", QLabel(devices_text))
        layout.addWidget(gpu_group)

        # System section
        sys_group = QGroupBox("System")
        sys_form = QFormLayout(sys_group)
        sys_form.addRow(
            "Total RAM:", QLabel(f"{self._capabilities.total_ram_gb:.1f} GB")
        )
        sys_form.addRow("CPU cores:", QLabel(str(self._capabilities.cpu_cores)))
        layout.addWidget(sys_group)

        # Reconstruction methods section
        methods_group = QGroupBox("Reconstruction Methods Available")
        methods_form = QFormLayout(methods_group)
        method_availability = [
            (ReconstructionMethod.METHOD_E, self._capabilities.can_run_method_e),
            (ReconstructionMethod.METHOD_D, self._capabilities.can_run_method_d),
            (ReconstructionMethod.METHOD_C, self._capabilities.can_run_method_c),
            (ReconstructionMethod.METHOD_CLOUD, True),
        ]
        for method, can_run in method_availability:
            reqs = MethodSelector.get_method_requirements(method)
            name = reqs.get("name", method.value)
            label = QLabel(_yes_no(can_run))
            label.setStyleSheet(
                "color: #4caf50;" if can_run else "color: #ff6666;"
            )
            methods_form.addRow(f"{name}:", label)
        layout.addWidget(methods_group)

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
