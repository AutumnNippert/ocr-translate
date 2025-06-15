import sys
import signal
from PySide6 import QtWidgets
from ui.overlay_window import OverlayWindow
from ui.main_window import MainWindow

def handle_sigint(*args):
    QtWidgets.QApplication.quit()

signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    # Parse CLI args
    show_window = "--window" in sys.argv
    show_overlay = "--overlay" in sys.argv

    # Default: show window if neither specified
    if not show_window and not show_overlay:
        show_window = True

    win = None
    overlay = None

    if show_window or show_overlay:
        win = MainWindow()
        if show_window:
            win.show()
        if show_overlay:
            overlay = OverlayWindow(
                win.get_top_words,
                lambda: (
                    win.screen_grabber.capture_x,
                    win.screen_grabber.capture_y,
                    win.screen_grabber.capture_w,
                    win.screen_grabber.capture_h,
                ),
            )
            overlay.show()

    app.exec()