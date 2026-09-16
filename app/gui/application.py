"""Main application entry point for GUI."""

import sys
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication
from app.gui.main_window import MainWindow
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


def setup_logging_gui(log_dir: str | Path = "logs") -> None:
    """Setup logging for GUI application."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir)


def main():
    """Main application entry point."""
    setup_logging_gui()
    
    logger.info("Starting Forensic Plate Enhancer GUI")
    
    app = QApplication(sys.argv)
    app.setApplicationName("Forensic Plate Enhancer")
    app.setApplicationVersion("1.0.0")
    
    window = MainWindow()
    window.show()
    
    logger.info("GUI application initialized")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
