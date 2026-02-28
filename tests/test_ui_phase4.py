"""
Tests for Phase 4 UI integration.

Covers:
- ControlPanel method selection properties (no display required)
- ProgressWidget stage label API
- MethodStatusWidget state machine
- HardwareInfoDialog construction (skipped without display)
- Wiring logic (selected_method, reconstruction_mode)

PySide6 widget tests are skipped when no display / QApplication is unavailable.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.reconstruction.method_selector import (
    HardwareCapabilities,
    MethodSelector,
    ReconstructionMethod,
)

# ---------------------------------------------------------------------------
# Try to boot a QApplication for widget tests; skip if unavailable.
# ---------------------------------------------------------------------------

_QAPP = None
_PYSIDE6_AVAILABLE = False

try:
    import sys
    from PySide6.QtWidgets import QApplication

    # Create exactly one QApplication for the whole test run.
    if QApplication.instance() is None:
        _QAPP = QApplication.instance() or QApplication(sys.argv[:1])
    else:
        _QAPP = QApplication.instance()
    _PYSIDE6_AVAILABLE = True
except Exception:
    pass

_requires_qt = unittest.skipUnless(
    _PYSIDE6_AVAILABLE, "PySide6 / display not available"
)


# ---------------------------------------------------------------------------
# MethodSelector logic (no Qt required)
# ---------------------------------------------------------------------------

class TestMethodSelectorLogic(unittest.TestCase):
    """Sanity-check MethodSelector.select_method used by the UI."""

    def _hw(self, vram: float = 8.0, has_cuda: bool = True) -> HardwareCapabilities:
        return HardwareCapabilities(
            has_cuda=has_cuda, has_mps=False, total_vram_gb=vram
        )

    def test_auto_chain_high_vram(self):
        hw = self._hw(vram=8.0)
        chain = MethodSelector.select_method(hw)
        self.assertIn(ReconstructionMethod.METHOD_E, chain)
        self.assertIn(ReconstructionMethod.METHOD_D, chain)
        self.assertIn(ReconstructionMethod.METHOD_C, chain)

    def test_user_preference_placed_first(self):
        hw = self._hw(vram=8.0)
        chain = MethodSelector.select_method(
            hw, user_preference=ReconstructionMethod.METHOD_D
        )
        self.assertEqual(chain[0], ReconstructionMethod.METHOD_D)

    def test_auto_chain_no_gpu(self):
        hw = self._hw(vram=0.0, has_cuda=False)
        chain = MethodSelector.select_method(hw)
        self.assertIn(ReconstructionMethod.METHOD_C, chain)
        # METHOD_E and METHOD_D require GPU — should not appear
        self.assertNotIn(ReconstructionMethod.METHOD_E, chain)
        self.assertNotIn(ReconstructionMethod.METHOD_D, chain)


# ---------------------------------------------------------------------------
# ControlPanel
# ---------------------------------------------------------------------------

@_requires_qt
class TestControlPanelMethodSelection(unittest.TestCase):
    """Tests for the updated ControlPanel method selection API."""

    def setUp(self):
        from ui.widgets.control_panel import ControlPanel

        # Patch hardware detection so tests don't depend on host GPU
        self._hw_no_gpu = HardwareCapabilities(
            has_cuda=False, has_mps=False, total_vram_gb=0.0
        )
        with patch.object(MethodSelector, "detect_hardware", return_value=self._hw_no_gpu):
            self.panel = ControlPanel()

    def test_default_mode_is_auto(self):
        self.assertEqual(self.panel.reconstruction_mode, "auto")

    def test_selected_method_is_none_for_auto(self):
        self.assertIsNone(self.panel.selected_method)

    def test_cloud_radio_sets_mode(self):
        self.panel._cloud_radio.setChecked(True)
        self.assertEqual(self.panel.reconstruction_mode, "cloud")

    def test_method_c_radio_sets_mode(self):
        self.panel._method_c_radio.setChecked(True)
        self.assertEqual(self.panel.reconstruction_mode, "method_c")

    def test_method_c_selected_method(self):
        self.panel._method_c_radio.setChecked(True)
        self.assertEqual(
            self.panel.selected_method, ReconstructionMethod.METHOD_C
        )

    def test_method_e_disabled_without_gpu(self):
        # Method E requires ≥6 GB VRAM — should be disabled with no GPU
        self.assertFalse(self.panel._method_e_radio.isEnabled())

    def test_method_d_disabled_without_gpu(self):
        # Method D requires ≥4 GB VRAM — should be disabled with no GPU
        self.assertFalse(self.panel._method_d_radio.isEnabled())

    def test_method_c_always_enabled(self):
        # Method C runs on CPU — always available
        self.assertTrue(self.panel._method_c_radio.isEnabled())

    def test_hardware_info_signal_exists(self):
        from PySide6.QtCore import Signal

        self.assertTrue(hasattr(self.panel, "hardware_info_requested"))

    def test_set_processing_disables_radios(self):
        self.panel.set_processing(True)
        self.assertFalse(self.panel._auto_radio.isEnabled())
        self.assertFalse(self.panel._cloud_radio.isEnabled())

    def test_set_processing_false_restores_available_radios(self):
        self.panel.set_processing(True)
        self.panel.set_processing(False)
        # Auto and cloud should be re-enabled
        self.assertTrue(self.panel._auto_radio.isEnabled())
        self.assertTrue(self.panel._cloud_radio.isEnabled())
        # Method E/D should stay disabled (no GPU)
        self.assertFalse(self.panel._method_e_radio.isEnabled())
        self.assertFalse(self.panel._method_d_radio.isEnabled())

    def test_method_e_enabled_with_high_vram(self):
        hw_gpu = HardwareCapabilities(
            has_cuda=True, has_mps=False, total_vram_gb=8.0
        )
        with patch.object(MethodSelector, "detect_hardware", return_value=hw_gpu):
            from ui.widgets.control_panel import ControlPanel

            panel = ControlPanel()
        self.assertTrue(panel._method_e_radio.isEnabled())
        self.assertTrue(panel._method_d_radio.isEnabled())


# ---------------------------------------------------------------------------
# ProgressWidget stage label
# ---------------------------------------------------------------------------

@_requires_qt
class TestProgressWidgetStageLabel(unittest.TestCase):
    """Tests for the new stage-label API added to ProgressWidget."""

    def setUp(self):
        from ui.widgets.progress_widget import ProgressWidget

        self.widget = ProgressWidget()

    def test_stage_label_hidden_by_default(self):
        self.assertFalse(self.widget._stage_label.isVisible())

    def test_set_stage_label_shows_text(self):
        self.widget.set_stage_label("Attempting Method D…")
        self.assertTrue(self.widget._stage_label.isVisible())
        self.assertEqual(self.widget._stage_label.text(), "Attempting Method D…")

    def test_set_stage_label_empty_hides(self):
        self.widget.set_stage_label("something")
        self.widget.set_stage_label("")
        self.assertFalse(self.widget._stage_label.isVisible())

    def test_clear_stage_label_hides(self):
        self.widget.set_stage_label("x")
        self.widget.clear_stage_label()
        self.assertFalse(self.widget._stage_label.isVisible())

    def test_set_complete_clears_stage_label(self):
        self.widget.set_stage_label("attempting something…")
        self.widget.set_complete()
        self.assertFalse(self.widget._stage_label.isVisible())

    def test_reset_clears_stage_label(self):
        self.widget.set_stage_label("x")
        self.widget.reset()
        self.assertFalse(self.widget._stage_label.isVisible())


# ---------------------------------------------------------------------------
# MethodStatusWidget
# ---------------------------------------------------------------------------

@_requires_qt
class TestMethodStatusWidget(unittest.TestCase):
    """Tests for MethodStatusWidget state transitions."""

    def setUp(self):
        from ui.widgets.method_status_widget import MethodStatusWidget

        self.widget = MethodStatusWidget()

    def test_initial_current_label_hidden(self):
        self.assertFalse(self.widget._current_label.isVisible())

    def test_set_current_method_shows_label(self):
        self.widget.set_current_method("Hybrid Photogrammetry")
        self.assertTrue(self.widget._current_label.isVisible())
        self.assertIn("Hybrid Photogrammetry", self.widget._current_label.text())

    def test_record_attempt_success(self):
        self.widget.set_current_method("Method C")
        self.widget.record_attempt("Method C", True)
        self.assertEqual(len(self.widget._attempts), 1)
        name, success = self.widget._attempts[0]
        self.assertEqual(name, "Method C")
        self.assertTrue(success)

    def test_record_attempt_failure(self):
        self.widget.set_current_method("Method E")
        self.widget.record_attempt("Method E", False)
        _, success = self.widget._attempts[0]
        self.assertFalse(success)

    def test_multiple_attempts_accumulate(self):
        self.widget.set_current_method("A")
        self.widget.record_attempt("A", False)
        self.widget.set_current_method("B")
        self.widget.record_attempt("B", True)
        self.assertEqual(len(self.widget._attempts), 2)

    def test_reset_clears_state(self):
        self.widget.set_current_method("A")
        self.widget.record_attempt("A", False)
        self.widget.reset()
        self.assertEqual(self.widget._attempts, [])
        self.assertFalse(self.widget._current_label.isVisible())


# ---------------------------------------------------------------------------
# HardwareInfoDialog
# ---------------------------------------------------------------------------

@_requires_qt
class TestHardwareInfoDialog(unittest.TestCase):
    """Tests for HardwareInfoDialog construction."""

    def _make_dialog(self, **kwargs) -> "HardwareInfoDialog":
        from ui.dialogs.hardware_info_dialog import HardwareInfoDialog

        hw = HardwareCapabilities(
            has_cuda=kwargs.get("has_cuda", False),
            has_mps=kwargs.get("has_mps", False),
            total_vram_gb=kwargs.get("total_vram_gb", 0.0),
            total_ram_gb=kwargs.get("total_ram_gb", 16.0),
            cpu_cores=kwargs.get("cpu_cores", 4),
            platform=kwargs.get("platform", "Linux"),
        )
        return HardwareInfoDialog(hw)

    def test_constructs_without_error(self):
        dialog = self._make_dialog()
        self.assertIsNotNone(dialog)

    def test_title_is_hardware_capabilities(self):
        dialog = self._make_dialog()
        self.assertEqual(dialog.windowTitle(), "Hardware Capabilities")

    def test_stores_capabilities(self):
        dialog = self._make_dialog(has_cuda=True, total_vram_gb=8.0)
        self.assertTrue(dialog._capabilities.has_cuda)
        self.assertAlmostEqual(dialog._capabilities.total_vram_gb, 8.0)

    def test_importable_from_dialogs_package(self):
        from ui.dialogs.hardware_info_dialog import HardwareInfoDialog  # noqa: F401


if __name__ == "__main__":
    unittest.main()
