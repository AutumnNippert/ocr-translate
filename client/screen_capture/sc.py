from PySide6 import QtWidgets, QtGui, QtCore
from Xlib import display as xdisplay
from Xlib.ext import shape, region
import sys

class TransparentOverlay(QtWidgets.QWidget):
    def __init__(self, screen_index=0):
        super().__init__()

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # Make it fullscreen on specified screen
        screen = QtGui.QGuiApplication.screens()[screen_index]
        self.setGeometry(screen.geometry())

        self._make_click_through()
        self.showFullScreen()

        # Timer to print cursor position every 50ms
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._print_cursor_pos)
        self.timer.start(50)

    def _make_click_through(self):
        wid = int(self.winId())
        dsp = xdisplay.Display()
        win = dsp.create_resource_object('window', wid)

        empty = region.create_region([])              # truly empty region

        shape.combine_region(
            dsp, win,
            shape.SK.Input,    # kind   (Input mask)
            shape.SO.Set,      # op     (replace)
            0, 0,              # x, y   offset
            empty
        )
        dsp.flush()
        print("[+] Click-through shape applied")


    def _print_cursor_pos(self):
        pos = QtGui.QCursor.pos()
        print(f"Mouse: {pos.x()}, {pos.y()}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    print("Qt backend:", QtGui.QGuiApplication.platformName())
    overlay = TransparentOverlay(screen_index=0)
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Exiting.")
        sys.exit(0)
