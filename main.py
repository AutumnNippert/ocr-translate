#!/usr/bin/env python3
"""
Live‑OCR‑Overlay  – deduplicated word list (20‑poll expiry)
=========================================================

A Qt‑based screen‑duplication tool that lists words currently visible on your
screen.  A word is *kept* until it has failed to re‑appear in **20 consecutive
OCR passes**, then it silently disappears from the list.

Changes in this revision
------------------------
* **Poll‑counter expiry** – we track how many OCR cycles a word has been MIA;
  after 20 misses (≈ 60 s with default settings) it’s dropped.
* `WORD_TTL` replaced by `WORD_MAX_MISSES`.
* Minor clean‑ups in the worker + prune logic.

Dependencies (unchanged)
------------------------
```bash
sudo pacman -S python-pyside6 tesseract tesseract-data-eng
pip install mss numpy pytesseract
```

Run with `python live_ocr_overlay_gui.py`.
"""

from __future__ import annotations
import sys, queue, threading, re
from typing import Dict, List, Tuple

import numpy as np
from mss import mss
import pytesseract
from PySide6 import QtCore, QtGui, QtWidgets
import cv2  # resize helper only

# ───────────────── CONFIG ─────────────────
LANG_OCR      = "deu"          # Tesseract language
CONF_MIN      = 85             # min confidence to keep a word
DOWNSCALE     = 1            # 0.5–0.7 gives decent speed/accuracy
OCR_INTERVAL  = 10              # feed every Nth captured frame to OCR
NUM_WORKERS   = 4              # parallel OCR threads
WORD_MAX_MISSES = 5           # remove word after N missed OCR polls
FONT_FAMILY   = "Noto Sans"    # bottom‑list font
FONT_SIZE_PT  = 12

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ───────────────── SCREEN‑GRAB THREAD ─────────────────
class ScreenGrabber(QtCore.QThread):
    frameCaptured = QtCore.Signal(np.ndarray)  # BGR frame

    def __init__(self, monitor: int = 1, parent=None):
        super().__init__(parent)
        self._monitor = monitor
        self._running = True

    def run(self):
        with mss() as sct:
            mon = sct.monitors[self._monitor]
            while self._running:
                raw = sct.grab(mon)
                frame = np.asarray(raw, dtype=np.uint8)[..., :3]  # BGR
                self.frameCaptured.emit(frame.copy())

    def stop(self):
        self._running = False
        self.wait()

# ───────────────── OCR WORKER THREAD ─────────────────
_TOKENISER = re.compile(r"[A-Za-zÀ-ÿ]+", re.UNICODE)

def cv2_resize(img: np.ndarray, fx=1.0, fy=1.0):
    return cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

class OCRWorker(QtCore.QThread):
    wordsFound = QtCore.Signal(list)  # list[str]

    def __init__(self, task_q: "queue.Queue[np.ndarray]", parent=None):
        super().__init__(parent)
        self.task_q = task_q
        self._running = True

    def run(self):
        while self._running:
            try:
                frame = self.task_q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.process(frame)
            self.task_q.task_done()

    def process(self, frame: np.ndarray):
        small = cv2_resize(frame, fx=DOWNSCALE, fy=DOWNSCALE)
        data = pytesseract.image_to_data(
            small, lang=LANG_OCR, config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
        conf = np.asarray(data["conf"], int)
        keep = conf >= CONF_MIN
        if not keep.any():
            return
        text_arr = np.asarray(data["text"])[keep]
        words: List[str] = []
        for token in text_arr:
            words.extend(_TOKENISER.findall(token.lower()))
        if words:
            self.wordsFound.emit(words)

    def stop(self):
        self._running = False
        self.wait()

# ───────────────── MAIN WINDOW ─────────────────
class MainWindow(QtWidgets.QMainWindow):
    """Tracks words and their consecutive‑miss counters."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live‑OCR‑Overlay (GUI)")
        self.resize(1000, 700)

        # layout
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central); vbox.setContentsMargins(0, 0, 0, 0)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.video_label, stretch=3)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setWordWrap(True)
        self.list_widget.setFont(QtGui.QFont(FONT_FAMILY, FONT_SIZE_PT))
        vbox.addWidget(self.list_widget, stretch=2)

        # state: word → miss_count (int)
        self.active_words: Dict[str, int] = {}

        # threads
        self._ocr_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=NUM_WORKERS)
        self.grabber = ScreenGrabber(); self.grabber.frameCaptured.connect(self.on_frame)
        self.grabber.start()

        self.ocr_workers = [OCRWorker(self._ocr_q) for _ in range(NUM_WORKERS)]
        for w in self.ocr_workers:
            w.wordsFound.connect(self.on_words)
            w.start()

        # timers
        self._frame_counter = 0
        self._fps_timer = QtCore.QElapsedTimer(); self._fps_timer.start()
        QtCore.QTimer.singleShot(1000, self.update_fps)

        self.prune_timer = QtCore.QTimer(self); self.prune_timer.timeout.connect(self.prune_words)
        self.prune_timer.start(500)  # every 0.5 s → ≈ 2 polls/s

    # ─── slots ───
    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        if self._frame_counter % OCR_INTERVAL == 0:
            try: self._ocr_q.put_nowait(frame.copy())
            except queue.Full: pass

        # display
        h, w = frame.shape[:2]
        qimg = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format_BGR888)
        pix  = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self._frame_counter += 1

    @QtCore.Slot(list)
    def on_words(self, words: list):
        # reset miss counter for every observed word
        for w in words:
            self.active_words[w] = 0

    # ─── maintenance ───
    def prune_words(self):
        # increment miss counts, drop those beyond limit
        remove: List[str] = []
        for w in list(self.active_words.keys()):
            self.active_words[w] += 1
            if self.active_words[w] >= WORD_MAX_MISSES:
                remove.append(w)
        for w in remove:
            del self.active_words[w]

        # refresh list widget
        self.list_widget.clear(); self.list_widget.addItems(sorted(self.active_words.keys()))

    def update_fps(self):
        elapsed = self._fps_timer.elapsed() / 1000
        fps = self._frame_counter / elapsed if elapsed else 0.0
        self.setWindowTitle(f"Live‑OCR‑Overlay – {fps:.0f} FPS | {len(self.active_words)} words tracked")
        self._frame_counter = 0; self._fps_timer.restart()
        QtCore.QTimer.singleShot(1000, self.update_fps)

    # ─── cleanup ───
    def closeEvent(self, e: QtGui.QCloseEvent):
        self.grabber.stop(); [w.stop() for w in self.ocr_workers]
        super().closeEvent(e)

# ───────────── ENTRY POINT ─────────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow(); mw.show()
    sys.exit(app.exec())
