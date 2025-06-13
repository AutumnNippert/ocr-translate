from PySide6 import QtWidgets
import sys
from ui.main_window import MainWindow
import signal
from PySide6.QtWidgets import QApplication

def handle_sigint(*args):
    QApplication.quit()

signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())