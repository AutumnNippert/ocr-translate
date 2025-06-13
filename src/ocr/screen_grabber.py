import time
import numpy as np
from PySide6 import QtCore
from mss import mss
from constants import CAPTURE_FPS_CAP

class ScreenGrabber(QtCore.QThread):
    frameCaptured = QtCore.Signal(np.ndarray)

    capture_x = 0
    capture_y = 0
    capture_w = 1920
    capture_h = 1080

    def __init__(self, id=1):
        super().__init__()
        self.id = id
        self._run = True
        with mss() as s:
            self.mon = s.monitors[id]
            self.capture_x = self.mon['left']
            self.capture_y = self.mon['top']
            self.capture_w = self.mon['width']
            self.capture_h = self.mon['height']
            self.setObjectName(f"ScreenGrabber-{id}")
            print(f"ScreenGrabber initialized for monitor {id}: {self.capture_x}, {self.capture_y}, {self.capture_w}, {self.capture_h}")

    def run(self):
        with mss() as s:
            mon = s.monitors[self.id]
            prev = 0
            while self._run:
                if time.time() - prev < 1 / CAPTURE_FPS_CAP:
                    time.sleep(.002)
                    continue
                prev = time.time()
                f = np.asarray(s.grab(mon), dtype=np.uint8)[..., :3].copy()
                self.frameCaptured.emit(f)

    def stop(self):
        self._run = False
        self.wait()