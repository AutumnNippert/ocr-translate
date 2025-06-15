from PySide6 import QtCore, QtGui, QtWidgets

class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, get_top_words_func):
        super().__init__()
        self.get_top_words = get_top_words_func  # Function to get [(word, gloss), ...]
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setWindowTitle("ScreenCapTranslate Overlay")
        self.resize(400, 200)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(40)  # ~25 FPS

    def update_overlay(self):
        # Move overlay near mouse, but not off screen
        mouse = QtGui.QCursor.pos()
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        x = min(mouse.x() + 24, screen.width() - self.width() - 10)
        y = min(mouse.y() + 24, screen.height() - self.height() - 10)
        self.move(x, y)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # Draw transparent rounded rectangle
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setBrush(QtGui.QColor(30, 30, 30, 180))  # semi-transparent dark
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect, 16, 16)

        # Draw words and definitions
        words = self.get_top_words()
        font = QtGui.QFont("Noto Sans", 12)
        painter.setFont(font)
        y = 32
        for word, gloss in words:
            painter.setPen(QtGui.QColor(255, 255, 255, 230))
            painter.drawText(24, y, word)
            y += 24
            painter.setPen(QtGui.QColor(180, 220, 255, 200))
            painter.drawText(36, y, gloss)
            y += 32

    # Optional: remove window from taskbar
    def showEvent(self, event):
        self.setWindowFlag(QtCore.Qt.Tool, True)
        super().showEvent(event)