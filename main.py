#!/usr/bin/env python3
"""
Live-OCR-Overlay – fixed-size word cards + cursor hover priority
==============================================================

**What changed**
* Light card theme (`#eee` on `#bbb` border)
* Fixed card size 160 × 60 px, four cards per row
* Hover bonus (`HOVER_BONUS`) pushes the line under the mouse cursor to the
  top-left corner

All detection parameters (down-scale, CLAHE, OEM 1, etc.) remain where you left
them.
"""

from __future__ import annotations
import sys, time, queue, threading, re
from typing import Dict, List, Tuple

import numpy as np
from mss import mss
import pytesseract
from PySide6 import QtCore, QtGui, QtWidgets
import cv2

# ───────────────── CONFIG ─────────────────
LANG_OCR        = "deu"
CONF_MIN        = 50
DOWNSCALE       = 0.9
OCR_INTERVAL    = 20
NUM_WORKERS     = 4
WORD_MAX_MISSES = 10
HOVER_BONUS     = 1000   # prominence bonus for hovered line

FONT_FAMILY     = "Noto Sans"
FONT_SIZE_PT    = 12
CARD_W, CARD_H  = 160, 60

PREPROCESS_CONTRAST = True
CLAHE_CLIP      = 2
CLAHE_GRID      = (16, 16)

TESS_CONFIG = "--oem 1 --psm 11"
CAPTURE_FPS_CAP   = 10
WORKER_IDLE_SLEEP = 0.02

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ───────── helpers ─────────
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{2,}+", re.UNICODE)

def resize(img: np.ndarray, fx=1.0, fy=1.0):
    return cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

def preprocess(img: np.ndarray) -> np.ndarray:
    if not PREPROCESS_CONTRAST:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return clahe.apply(gray)

# ───────── threads ─────────
class ScreenGrabber(QtCore.QThread):
    frameCaptured = QtCore.Signal(np.ndarray)

    def __init__(self, monitor_id: int = 1):
        super().__init__()
        self.monitor_id = monitor_id
        self._running = True
        with mss() as sct:
            self.mon = sct.monitors[self.monitor_id]  # for cursor mapping

    def run(self):
        with mss() as sct:
            mon = sct.monitors[self.monitor_id]
            last = 0.0
            while self._running:
                if time.time() - last < 1 / CAPTURE_FPS_CAP:
                    time.sleep(0.002)
                    continue
                last = time.time()
                frame = np.asarray(sct.grab(mon), dtype=np.uint8)[..., :3].copy()
                self.frameCaptured.emit(frame)

    def stop(self):
        self._running = False
        self.wait()


class OCRWorker(QtCore.QThread):
    """Emit a list of (text, bbox) for each detected line."""

    linesFound = QtCore.Signal(list)  # list[Tuple[str, bbox]]

    def __init__(self, task_q: "queue.Queue[np.ndarray]"):
        super().__init__()
        self.task_q = task_q
        self._running = True

    def run(self):
        while self._running:
            try:
                frame = self.task_q.get(timeout=WORKER_IDLE_SLEEP)
            except queue.Empty:
                continue
            self.process(frame)
            self.task_q.task_done()

    def process(self, frame: np.ndarray):
        small = resize(frame, fx=DOWNSCALE, fy=DOWNSCALE)
        pre   = preprocess(small)
        data  = pytesseract.image_to_data(pre, lang=LANG_OCR,
                                         config=TESS_CONFIG,
                                         output_type=pytesseract.Output.DICT)
        conf = np.asarray(data["conf"], int)
        keep = conf >= CONF_MIN
        if not keep.any():
            return

        blk = np.asarray(data["block_num"])[keep]
        ln  = np.asarray(data["line_num"])[keep]
        txt = np.asarray(data["text"     ])[keep]
        lft = np.asarray(data["left"     ])[keep] / DOWNSCALE
        top = np.asarray(data["top"      ])[keep] / DOWNSCALE
        wid = np.asarray(data["width"    ])[keep] / DOWNSCALE
        hgt = np.asarray(data["height"   ])[keep] / DOWNSCALE

        grouped: Dict[Tuple[int,int], List[int]] = {}
        for i, (b, l) in enumerate(zip(blk, ln)):
            if txt[i].strip():
                grouped.setdefault((b, l), []).append(i)

        out: List[Tuple[str, Tuple[int,int,int,int]]] = []
        for idxs in grouped.values():
            sentence = " ".join(txt[i] for i in idxs).strip()
            if not sentence:
                continue
            x0 = int(lft[idxs].min())
            y0 = int(top[idxs].min())
            x1 = int((lft[idxs] + wid[idxs]).max())
            y1 = int((top[idxs] + hgt[idxs]).max())
            out.append((sentence, (x0, y0, x1, y1)))

        if out:
            self.linesFound.emit(out)

    def stop(self):
        self._running = False
        self.wait()

# ───────── GUI ─────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live-OCR-Overlay")
        self.resize(1000, 700)

        # ── layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.video, 3)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_container)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setSpacing(10)
        self.scroll.setWidget(self.grid_container)
        vbox.addWidget(self.scroll, 2)

        # ── state
        # word → {freq:int, miss:int, bbox:(x0,y0,x1,y1)}
        self.words: Dict[str, Dict[str, int | Tuple[int,int,int,int]]] = {}
        self.task_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=NUM_WORKERS)

        # ── threads
        self.grabber = ScreenGrabber()
        self.grabber.frameCaptured.connect(self.on_frame)
        self.grabber.start()
        self.mon_geom = self.grabber.mon

        self.workers = [OCRWorker(self.task_q) for _ in range(NUM_WORKERS)]
        for w in self.workers:
            w.linesFound.connect(self.on_lines)
            w.start()

        # ── timers
        self.frame_counter = 0
        self.fps_timer = QtCore.QElapsedTimer(); self.fps_timer.start()
        QtCore.QTimer.singleShot(1000, self.update_fps)

        self.prune_timer = QtCore.QTimer(self)
        self.prune_timer.timeout.connect(self.prune)
        self.prune_timer.start(500)

    # ── slots
    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        if self.frame_counter % OCR_INTERVAL == 0:
            try:
                self.task_q.put_nowait(frame)
            except queue.Full:
                pass

        h, w = frame.shape[:2]
        qimg = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format_BGR888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.frame_counter += 1

    @QtCore.Slot(list)
    def on_lines(self, lines: List[Tuple[str, Tuple[int,int,int,int]]]):
        for text, bbox in lines:
            info = self.words.setdefault(text, {"freq": 0, "miss": 0, "bbox": bbox})
            info["freq"] += 1
            info["miss"] = 0
            info["bbox"] = bbox
        self.refresh_grid()

    # ── maintenance
    def prune(self):
        for w in list(self.words):
            self.words[w]["miss"] += 1
            if self.words[w]["miss"] >= WORD_MAX_MISSES:
                del self.words[w]
        self.refresh_grid()

    # ── UI helpers
    def refresh_grid(self):
        # cursor in monitor space
        cur_global = QtGui.QCursor.pos()
        cur_x = cur_global.x() - self.mon_geom["left"]
        cur_y = cur_global.y() - self.mon_geom["top"]

        scored: List[Tuple[int, str]] = []
        for word, meta in self.words.items():
            x0, y0, x1, y1 = meta["bbox"]
            hovered = x0 <= cur_x <= x1 and y0 <= cur_y <= y1
            score = meta["freq"] + (HOVER_BONUS if hovered else 0)
            scored.append((score, word))
        scored.sort(reverse=True)

        # clear grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = 4
        for idx, (_, word) in enumerate(scored):
            row, col = divmod(idx, cols)
            card = QtWidgets.QFrame()
            card.setFixedSize(CARD_W, CARD_H)
            card.setStyleSheet("background:#eee;border:1px solid #bbb;border-radius:6px;")

            label = QtWidgets.QLabel(word.upper(), alignment=QtCore.Qt.AlignCenter)
            label.setFont(QtGui.QFont(FONT_FAMILY, FONT_SIZE_PT))

            layout = QtWidgets.QVBoxLayout(card)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.addWidget(label)

            self.grid.addWidget(card, row, col)

        self.grid_container.adjustSize()

    def update_fps(self):
        elapsed = max(1, self.fps_timer.elapsed() / 1000)
        fps = self.frame_counter / elapsed
        self.setWindowTitle(f"Live-OCR-Overlay – {fps:.0f} FPS | {len(self.words)} words")
        self.frame_counter = 0
        self.fps_timer.restart()
        QtCore.QTimer.singleShot(1000, self.update_fps)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.grabber.stop()
        for w in self.workers:
            w.stop()
        super().closeEvent(event)

# ───────── entry point ─────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())