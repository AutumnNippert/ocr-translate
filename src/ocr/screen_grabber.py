import time
import numpy as np
from PySide6 import QtCore
from mss import mss
from constants import CAPTURE_FPS_CAP

class ScreenGrabber(QtCore.QThread):
    frameCaptured = QtCore.Signal(np.ndarray)

    def __init__(self, id=1):
        super().__init__()
        self.id = id
        self._run = True
        with mss() as s:
            self.mon = s.monitors[id]

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