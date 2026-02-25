"""
Image2STL - Main Entry Point
"""
import sys
import os
import logging
from pathlib import Path

try:
    import PySide6

    _qt_plugins = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    os.environ.setdefault("QT_PLUGIN_PATH", str(_qt_plugins))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_qt_plugins / "platforms"))
except Exception:
    pass

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from ui.main_window import MainWindow
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("image2stl.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point"""
    try:
        qt_plugin_root = os.environ.get("QT_PLUGIN_PATH")
        if qt_plugin_root:
            QCoreApplication.setLibraryPaths([qt_plugin_root])
        app = QApplication(sys.argv)
        app.setApplicationName("Image2STL")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("Image2STL")

        icon_path = Path(__file__).parent / "ui" / "resources" / "icons" / "app_icon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))

        config = Config()

        stylesheet_path = Path(__file__).parent / "ui" / "resources" / "styles.qss"
        if stylesheet_path.exists():
            with open(stylesheet_path, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())

        window = MainWindow(config)
        window.show()

        logger.info("Application started successfully")

        sys.exit(app.exec())

    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
