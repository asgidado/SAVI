"""
SAVI v0.0.1 — Iris Tracking Core
Entry point: python main.py
"""
import os
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import sys
import logging
from PySide6.QtWidgets import QApplication
from savi.ui.tracker_window import TrackerWindow

# Configure basic logging to terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TrackerWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
