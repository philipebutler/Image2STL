"""
Progress widget - progress bar with status text, time warning, and error display.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QProgressBar,
    QLabel,
)


class ProgressWidget(QWidget):
    """Displays reconstruction progress, time warnings, and error messages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_progress(self, fraction: float, status: str, estimated_seconds: Optional[int] = None):
        """Update the progress bar and status text.

        Args:
            fraction: 0.0–1.0 progress value.
            status: Short status message.
            estimated_seconds: Optional estimated seconds remaining.
        """
        self._bar.setValue(int(fraction * 100))
        self._status_label.setText(status)

        if estimated_seconds is not None and estimated_seconds > 600:
            minutes = estimated_seconds // 60
            self._warning_label.setText(
                f"⚠  Estimated processing time exceeds 10 minutes ({minutes} min remaining)"
            )
            self._warning_label.setVisible(True)
        else:
            self._warning_label.setVisible(False)

    def set_complete(self):
        """Mark progress as complete."""
        self._bar.setValue(100)
        self._status_label.setText("Complete")
        self._warning_label.setVisible(False)

    def show_error(self, message: str, suggestion: str = ""):
        """Display an error message below the progress bar."""
        self._error_title.setText(message)
        self._error_suggestion.setText(suggestion)
        self._error_title.setVisible(True)
        self._error_suggestion.setVisible(bool(suggestion))
        self._error_frame.setVisible(True)

    def show_info(self, message: str):
        """Display an informational message below the progress bar."""
        self._info_label.setText(message)
        self._info_label.setVisible(bool(message))

    def clear_info(self):
        """Hide the informational message line."""
        self._info_label.setVisible(False)

    def clear_error(self):
        """Hide the error display."""
        self._error_frame.setVisible(False)

    def reset(self):
        """Reset to initial state."""
        self._bar.setValue(0)
        self._status_label.setText("")
        self._warning_label.setVisible(False)
        self._info_label.setVisible(False)
        self._error_frame.setVisible(False)

    # ------------------------------------------------------------------
    # Internal UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # Progress row
        progress_row = QHBoxLayout()
        self._bar = QProgressBar()
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setTextVisible(False)
        progress_row.addWidget(self._bar, 1)

        self._status_label = QLabel()
        self._status_label.setMinimumWidth(220)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        progress_row.addWidget(self._status_label)
        layout.addLayout(progress_row)

        # Time warning
        self._warning_label = QLabel()
        self._warning_label.setStyleSheet("color: #ffc000; background: rgba(255,192,0,0.12); padding: 4px 8px; border-radius: 4px;")
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

        # Informational summary display
        self._info_label = QLabel()
        self._info_label.setStyleSheet("color: #9fc5ff; background: rgba(100,149,237,0.12); padding: 4px 8px; border-radius: 4px;")
        self._info_label.setWordWrap(True)
        self._info_label.setVisible(False)
        layout.addWidget(self._info_label)

        # Error display
        self._error_frame = QWidget()
        self._error_frame.setStyleSheet("background: rgba(255,68,68,0.12); border-radius: 4px; padding: 4px;")
        error_layout = QVBoxLayout(self._error_frame)
        error_layout.setContentsMargins(8, 4, 8, 4)
        error_layout.setSpacing(2)

        self._error_title = QLabel()
        self._error_title.setStyleSheet("color: #ff6666; font-weight: bold;")
        error_layout.addWidget(self._error_title)

        self._error_suggestion = QLabel()
        self._error_suggestion.setStyleSheet("color: #ff9999;")
        self._error_suggestion.setWordWrap(True)
        error_layout.addWidget(self._error_suggestion)

        self._error_frame.setVisible(False)
        layout.addWidget(self._error_frame)
