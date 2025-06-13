#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-OCR-Overlay – fast ROI + deep OCR (fixed, 14 Jun 2025)
"""

from __future__ import annotations
import sys, time, queue, re, logging
from typing import Dict, List, Tuple
import numpy as np; import cv2; from mss import mss
from PySide6 import QtCore, QtGui, QtWidgets; import pytesseract

# ───── debug switch ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
DBG = True
# ───────────────── CONFIG ──────────────────────────────────────
LANG_OCR         = "deu+eng"
CONF_MIN_FAST    = 55
CONF_MIN_DEEP    = 60
FAST_SCALE       = 0.7          # ← NEW  (0.5 ⇒ 2160 p → 1080 p)
UPSCALE_FACTOR   = 2.0          # deep pass enlargement
OCR_INTERVAL     = 30
ROIS_PER_FRAME   = 30           # max ROIs per frame
NUM_WORKERS      = 3
WORD_MAX_MISSES  = 20
HOVER_BONUS      = 1000
FONT_FAMILY      = "Noto Sans"; FONT_SIZE_PT = 12

TESS_FAST_CFG    = "--oem 1 --psm 6"
TESS_DEEP_CFG    = "--oem 1 --psm 7"

CAPTURE_FPS_CAP  = 10
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
_WORD_RE         = re.compile(r"[A-Za-zÀ-ÿ]{2,}", re.UNICODE)

# ───── helper: non-max suppression ─────────────────────────────
def nms(rois: List[Tuple[int,int,int,int]], thr: float = 0.3)->List[Tuple[int,int,int,int]]:
    if not rois: return []
    b = np.asarray(rois,float); x1,y1,x2,y2=b.T; area=(x2-x1+1)*(y2-y1+1)
    idx = area.argsort()[::-1]; keep=[]
    while idx.size:
        i=idx[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[idx[1:]]); yy1=np.maximum(y1[i],y1[idx[1:]])
        xx2=np.minimum(x2[i],x2[idx[1:]]); yy2=np.minimum(y2[i],y2[idx[1:]])
        w=np.maximum(0,xx2-xx1+1); h=np.maximum(0,yy2-yy1+1)
        iou=(w*h)/(area[i]+area[idx[1:]]-w*h)
        idx=idx[1:][iou<thr]
    return [tuple(map(int,b[i])) for i in keep]

# ───── preprocessing fns ───────────────────────────────────────
def preprocess_fast(img: np.ndarray)->np.ndarray:
    """Fast, minimal pre-processing (gray only)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess_deep(gray: np.ndarray)->np.ndarray:
    """Blur + adaptive threshold for tiny fonts."""
    gray=cv2.GaussianBlur(gray,(3,3),0)
    return cv2.adaptiveThreshold(gray,255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,11,2)

# ───── threads ─────────────────────────────────────────────────
class ScreenGrabber(QtCore.QThread):
    frameCaptured = QtCore.Signal(np.ndarray)
    def __init__(self,id=1):
        super().__init__(); self.id=id; self._run=True
        with mss() as s:self.mon=s.monitors[id]
    def run(self):
        with mss() as s:
            mon=s.monitors[self.id]; prev=0
            while self._run:
                if time.time()-prev<1/CAPTURE_FPS_CAP: time.sleep(.002); continue
                prev=time.time()
                f=np.asarray(s.grab(mon),dtype=np.uint8)[...,:3].copy()
                self.frameCaptured.emit(f)
    def stop(self): self._run=False; self.wait()

class OCRWorker(QtCore.QThread):
    linesFound = QtCore.Signal(list)
    def __init__(self,q): super().__init__(); self.q=q; self._run=True
    def run(self):
        while self._run:
            try:f=self.q.get(timeout=.02)
            except queue.Empty: continue
            self.process(f); self.q.task_done()
    def stop(self): self._run=False; self.wait()

    # ---------- OCR pipeline ----------
    def _fast_rois(self,img)->List[Tuple[int,int,int,int]]:
        small=cv2.resize(img,(0,0),fx=FAST_SCALE,fy=FAST_SCALE)
        gray=preprocess_fast(small)
        d=pytesseract.image_to_data(gray,lang=LANG_OCR,config=TESS_FAST_CFG,
                                    output_type=pytesseract.Output.DICT)
        conf=np.asarray(d["conf"],int); keep=conf>=CONF_MIN_FAST
        if not keep.any(): return []
        lft=np.asarray(d["left"])[keep]/FAST_SCALE
        top=np.asarray(d["top"] )[keep]/FAST_SCALE
        wid=np.asarray(d["width"])[keep]/FAST_SCALE
        hgt=np.asarray(d["height"])[keep]/FAST_SCALE
        rois=[(int(x),int(y),int(x+w),int(y+h))
              for x,y,w,h in zip(lft,top,wid,hgt)]
        rois=nms(rois,0.3)[:ROIS_PER_FRAME]
        self.parent().last_roi_count = len(rois)   # notify GUI for title-bar
        return rois

    def _deep_words(self, img, roi):
        """Return refined words inside ROI – with tiny-retry."""
        x0, y0, x1, y1 = roi; pad = 4
        x0 = max(x0-pad, 0); y0 = max(y0-pad, 0)
        x1 = min(x1+pad, img.shape[1]-1); y1 = min(y1+pad, img.shape[0]-1)

        def ocr_pass(up_scale: float, psm_cfg: str, conf_min: int):
            crop = img[y0:y1, x0:x1]
            big  = cv2.resize(crop, (0, 0), fx=up_scale, fy=up_scale)
            g    = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            g    = cv2.dilate(g, np.ones((1,2), np.uint8))     # bridge thin strokes
            bw   = preprocess_deep(g)

            data = pytesseract.image_to_data(
                bw, lang=LANG_OCR,
                config=f"--oem 1 {psm_cfg}",
                output_type=pytesseract.Output.DICT,
            )
            conf = np.asarray(data["conf"], int)
            keep = conf >= conf_min
            if not keep.any():
                return []

            txt = np.asarray(data["text"] )[keep]
            lft = np.asarray(data["left"] )[keep] / up_scale + x0
            top = np.asarray(data["top"]  )[keep] / up_scale + y0
            wid = np.asarray(data["width"])[keep] / up_scale
            hgt = np.asarray(data["height"])[keep] / up_scale

            return [
                (t.strip().upper(),
                 (int(x), int(y), int(x+w), int(y+h)))
                for t, x, y, w, h in zip(txt, lft, top, wid, hgt)
                if len(t.strip()) > 1 and _WORD_RE.search(t)
            ]

        # first try: ×2, psm 7
        words = ocr_pass(UPSCALE_FACTOR, "--psm 7", CONF_MIN_DEEP)
        if words:
            return words

        # tiny-retry: ×3, psm 11, lower conf
        tiny_words = ocr_pass(3.0, "--psm 11", 50)
        if DBG and tiny_words:
            logging.info(f"        tiny-retry hit {len(tiny_words)} word(s)")
        return tiny_words

    def process(self,frame):
        t0=time.perf_counter()
        rois=self._fast_rois(frame)
        t_fast=(time.perf_counter()-t0)*1000
        if DBG: logging.info(f"[OCR] {len(rois)} ROI  fast {t_fast:.1f} ms",)
        if not rois: return
        words=[]
        for r in rois: words+=self._deep_words(frame,r)
        t_tot=(time.perf_counter()-t0)*1000
        if DBG: logging.info(f"        total {t_tot:.1f} ms  words {len(words)}")
        if words:self.linesFound.emit(words)

# ───── GUI & entry (unchanged from previous answer) ─────
# ... (same MainWindow, refresh_list, etc. as in previous message) ...

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
