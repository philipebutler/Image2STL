"""
Method status widget - shows which reconstruction method is running and attempt history.
"""
from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)


class MethodStatusWidget(QWidget):
    """Displays the active reconstruction method and the attempt history.

    Usage::

        widget.set_current_method("Hybrid Photogrammetry")   # method started
        widget.record_attempt("Hybrid Photogrammetry", False) # method failed
        widget.set_current_method("Dust3R Multi-View")        # next attempt
        widget.record_attempt("Dust3R Multi-View", True)      # success
        widget.reset()                                         # clear all
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._attempts: List[Tuple[str, bool]] = []
        self._build_ui()
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_current_method(self, method_name: str):
        """Show that *method_name* is now being attempted."""
        self._current_label.setText(f"⟳  {method_name}")
        self._current_label.setStyleSheet(
            "color: #9fc5ff; font-weight: bold; font-size: 12px;"
        )
        self._current_label.setVisible(True)

    def record_attempt(self, method_name: str, success: bool):
        """Record the outcome of a completed attempt and update the history row."""
        self._attempts.append((method_name, success))
        self._refresh_history()
        if success:
            self._current_label.setText(f"✓  {method_name}")
            self._current_label.setStyleSheet(
                "color: #4caf50; font-weight: bold; font-size: 12px;"
            )
        else:
            self._current_label.setText(f"✗  {method_name} — trying next…")
            self._current_label.setStyleSheet("color: #ff9999; font-size: 12px;")

    def reset(self):
        """Clear current method label and attempt history."""
        self._attempts = []
        self._current_label.setText("")
        self._current_label.setVisible(False)
        self._refresh_history()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._current_label = QLabel()
        self._current_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self._current_label)

        self._history_widget = QWidget()
        self._history_layout = QHBoxLayout(self._history_widget)
        self._history_layout.setContentsMargins(0, 0, 0, 0)
        self._history_layout.setSpacing(6)
        self._history_layout.addStretch(1)
        layout.addWidget(self._history_widget)

    def _refresh_history(self):
        """Rebuild attempt-history badges from ``self._attempts``."""
        # Remove all badge widgets (keep the trailing stretch item)
        while self._history_layout.count() > 1:
            item = self._history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for name, success in self._attempts:
            icon = "✓" if success else "✗"
            badge = QLabel(f"{icon}  {name}")
            color = "#4caf50" if success else "#ff6666"
            badge.setStyleSheet(
                f"color: {color}; background: rgba(0,0,0,0.20); "
                "border-radius: 3px; padding: 1px 6px; font-size: 11px;"
            )
            # Insert before the trailing stretch item (always at index count()-1)
            self._history_layout.insertWidget(
                self._history_layout.count() - 1, badge
            )
