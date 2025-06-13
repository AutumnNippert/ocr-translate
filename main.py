#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-OCR-Overlay – two-stage OCR  (13 Jun 2025)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stage A  quick column scan  (psm 6) → candidate line ROIs
Stage B  per-ROI deep scan (psm 7) → final words

Left-aligned priority list ordered by sightings + cursor hover.
"""

from __future__ import annotations
import sys, time, queue, threading, re
from typing import Dict, List, Tuple

import numpy as np
from mss import mss
import pytesseract
from PySide6 import QtCore, QtGui, QtWidgets
import cv2

import logging, time as _time
logging.basicConfig(level=logging.INFO, format="%(message)s")
DBG = True           # flip to False to silence debug prints

def nms(rois: List[Tuple[int,int,int,int]], thresh: float = 0.3) -> List[Tuple[int,int,int,int]]:
    """Non-maximum suppression for axis-aligned rectangles."""
    if not rois:
        return []
    boxes = np.asarray(rois, dtype=float)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < thresh]
    return [tuple(map(int, boxes[i])) for i in keep]


# ───────────────── CONFIG ─────────────────
LANG_OCR         = "eng"
CONF_MIN_FAST    = 65          # stage A
CONF_MIN_DEEP    = 60          # stage B (after up-scale)
DOWNSCALE        = 1.0         # stage A scale
UPSCALE_FACTOR   = 2.0         # stage B enlargement
OCR_INTERVAL     = 10
NUM_WORKERS      = 4
WORD_MAX_MISSES  = 30
HOVER_BONUS      = 1000

FONT_FAMILY      = "Noto Sans"
FONT_SIZE_PT     = 12

CLAHE_CLIP       = 0           # 0 ⇒ CLAHE off
CLAHE_GRID       = (16, 16)

TESS_FAST_CFG    = "--oem 1 --psm 6"   # column of multiple lines
TESS_DEEP_CFG    = "--oem 1 --psm 7"   # single refined line

CAPTURE_FPS_CAP  = 10
WORKER_IDLE_SLEEP= 0.02

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{2,}", re.UNICODE)
clahe    = cv2.createCLAHE(clipLimit=max(CLAHE_CLIP, 2), tileGridSize=CLAHE_GRID)

# ───────── helpers ─────────
def resize(img: np.ndarray, fx=1.0, fy=1.0):
    return cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

def preprocess(gray: np.ndarray) -> np.ndarray:
    if CLAHE_CLIP:
        gray = clahe.apply(gray)
    # adaptive threshold for deep (small) text
    return cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# ───────── threads ─────────
class ScreenGrabber(QtCore.QThread):
    frameCaptured = QtCore.Signal(np.ndarray)
    def __init__(self, monitor_id=1):
        super().__init__(); self.monitor_id=monitor_id; self._running=True
        with mss() as sct: self.mon = sct.monitors[monitor_id]
    def run(self):
        with mss() as sct:
            mon = sct.monitors[self.monitor_id]; last=0
            while self._running:
                if time.time()-last < 1/CAPTURE_FPS_CAP: time.sleep(0.002); continue
                last=time.time()
                frm = np.asarray(sct.grab(mon), dtype=np.uint8)[..., :3].copy()
                self.frameCaptured.emit(frm)
    def stop(self): self._running=False; self.wait()

class OCRWorker(QtCore.QThread):
    linesFound = QtCore.Signal(list)   # list[(text, bbox)]
    def __init__(self, q: "queue.Queue[np.ndarray]"):
        super().__init__(); self.q=q; self._run=True
    def run(self):
        while self._run:
            try: frm = self.q.get(timeout=WORKER_IDLE_SLEEP)
            except queue.Empty: continue
            self.process(frm); self.q.task_done()
    def stop(self): self._run=False; self.wait()

    # -------- two-stage OCR --------
    def _fast_scan(self, img: np.ndarray) -> List[Tuple[int,int,int,int]]:
        """Return coarse ROIs (x0,y0,x1,y1) for each likely text-line."""
        small = resize(img, fx=DOWNSCALE, fy=DOWNSCALE)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        data  = pytesseract.image_to_data(
            gray, lang=LANG_OCR, config=TESS_FAST_CFG,
            output_type=pytesseract.Output.DICT,
        )
        conf = np.asarray(data["conf"], int)
        keep = conf >= CONF_MIN_FAST
        if not keep.any():
            return []

        lft = np.asarray(data["left"] )[keep] / DOWNSCALE
        top = np.asarray(data["top"]  )[keep] / DOWNSCALE
        wid = np.asarray(data["width"])[keep] / DOWNSCALE
        hgt = np.asarray(data["height"])[keep] / DOWNSCALE

        rois = []
        for x, y, w, h in zip(lft, top, wid, hgt):
            rois.append((int(x), int(y), int(x+w), int(y+h)))
        
            # existing rois list constructed above
        rois = nms(rois, 0.3)                # merge overlaps
        rois = sorted(rois, key=lambda r: (r[1], r[0]))[:30]   # keep at most 30
        return rois

    def _deep_scan_roi(self, full_img: np.ndarray,
                       roi: Tuple[int,int,int,int]) -> list[Tuple[str, Tuple[int,int,int,int]]]:
        """Return refined words inside this ROI."""
        x0,y0,x1,y1 = roi
        pad = 4
        x0 = max(x0-pad,0); y0 = max(y0-pad,0)
        x1 = min(x1+pad, full_img.shape[1]-1)
        y1 = min(y1+pad, full_img.shape[0]-1)

        crop = full_img[y0:y1, x0:x1]
        big  = resize(crop, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        bw   = preprocess(gray)                  # adaptive

        data = pytesseract.image_to_data(
            bw, lang=LANG_OCR, config=TESS_DEEP_CFG,
            output_type=pytesseract.Output.DICT,
        )
        conf = np.asarray(data["conf"], int)
        keep = conf >= CONF_MIN_DEEP
        if not keep.any():
            return []

        txt = np.asarray(data["text"] )[keep]
        lft = np.asarray(data["left"] )[keep] / UPSCALE_FACTOR + x0
        top = np.asarray(data["top"]  )[keep] / UPSCALE_FACTOR + y0
        wid = np.asarray(data["width"])[keep] / UPSCALE_FACTOR
        hgt = np.asarray(data["height"])[keep] / UPSCALE_FACTOR

        out = []
        for t,x,y,w,h in zip(txt,lft,top,wid,hgt):
            tok = t.strip()
            if len(tok)<=1 or not _WORD_RE.search(tok):
                continue
            out.append((tok.upper(), (int(x),int(y),int(x+w),int(y+h))))
        return out

    # -------- high-level process --------
    def process(self, frame: np.ndarray):
        t0 = _time.perf_counter()

        rois = self._fast_scan(frame)
        t_fast = (_time.perf_counter() - t0) * 1000   # ms

        if not rois:
            if DBG:
                logging.info(f"[OCR] 0 ROI  (fast {t_fast:.1f} ms)")
            return

        words: list[Tuple[str, Tuple[int,int,int,int]]] = []
        for roi in rois:
            words.extend(self._deep_scan_roi(frame, roi))
        t_total = (_time.perf_counter() - t0) * 1000

        # pass ROI count up so UI can show it
        self.parent().last_roi_count = len(rois)

        if DBG:
            logging.info(f"[OCR] {len(rois)} ROI  "
                        f"fast {t_fast:.1f} ms  total {t_total:.1f} ms  "
                        f"words {len(words)}")

        if words:
            self.linesFound.emit(words)


# ───────── GUI (priority list) ─────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live-OCR-Overlay")
        self.resize(1000, 700)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central); vbox.setContentsMargins(0,0,0,0)

        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.video, 3)

        self.list = QtWidgets.QListWidget()
        self.list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setStyleSheet("QListWidget{padding:4px;} QListWidget::item{margin:2px 0;}")
        vbox.addWidget(self.list, 2)

        self.words: Dict[str, Dict] = {}
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=NUM_WORKERS)

        self.grabber = ScreenGrabber(); self.grabber.frameCaptured.connect(self.on_frame)
        self.grabber.start(); self.mon = self.grabber.mon

        self.workers = [OCRWorker(self.q) for _ in range(NUM_WORKERS)]
        for w in self.workers:
            w.setParent(self)                # allow w.parent() lookup
            w.linesFound.connect(self.on_lines)
            w.start()

        self.last_roi_count = 0
        self.fc = 0; self.t = QtCore.QElapsedTimer(); self.t.start()
        QtCore.QTimer.singleShot(1000, self.update_fps)
        self.prune_timer = QtCore.QTimer(self); self.prune_timer.timeout.connect(self.prune); self.prune_timer.start(500)

    # -------- slots --------
    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame):
        if self.fc % OCR_INTERVAL == 0:
            try: self.q.put_nowait(frame.copy())
            except queue.Full: pass

        h,w = frame.shape[:2]
        img = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format_BGR888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.fc += 1

    @QtCore.Slot(list)
    def on_lines(self, lines):
        for word, bbox in lines:
            meta = self.words.setdefault(word, {"freq":0, "miss":0, "bbox":bbox})
            meta["freq"] += 1
            meta["miss"] = 0
            meta["bbox"] = bbox
        self.refresh_list()

    def prune(self):
        for w in list(self.words):
            self.words[w]["miss"] += 1
            if self.words[w]["miss"] >= WORD_MAX_MISSES:
                del self.words[w]
        self.refresh_list()

    def refresh_list(self):
        cur = QtGui.QCursor.pos()
        cx = cur.x() - self.mon["left"]; cy = cur.y() - self.mon["top"]
        scored = []
        for word, meta in self.words.items():
            score = meta["freq"]
            x1,y1,x2,y2 = meta["bbox"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                score += HOVER_BONUS
            scored.append((score, word))
        scored.sort(reverse=True)
        self.list.clear()
        for _,word in scored:
            it = QtWidgets.QListWidgetItem(word)
            it.setFont(QtGui.QFont(FONT_FAMILY, FONT_SIZE_PT))
            self.list.addItem(it)
            if DBG and scored:
                logging.info("Top words: " +
                            ", ".join(f"{w}({s})" for s, w in scored[:8]))
        

    def update_fps(self):
        fps = self.fc / max(1, self.t.elapsed()/1000)
        self.setWindowTitle(
            f"Live-OCR-Overlay  |  {fps:.0f} FPS  |  {len(self.words)} words  "
            f"|  ROI {self.last_roi_count}"
        )
        self.fc = 0; self.t.restart(); QtCore.QTimer.singleShot(1000, self.update_fps)


    def closeEvent(self,e):
        self.grabber.stop(); [w.stop() for w in self.workers]; super().closeEvent(e)

# ───────── entry point ─────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())
