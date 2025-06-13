from PySide6 import QtCore, QtGui, QtWidgets
import html
import queue
from typing import Dict, List
import numpy as np
from translation import translate
from constants import (
    NUM_WORKERS, WORD_MAX_MISSES, HOVER_BONUS, FONT_FAMILY, FONT_SIZE_PT, OCR_INTERVAL
)
from ocr.screen_grabber import ScreenGrabber
from ocr.tesseract_ocr_worker import OCRWorker
import concurrent.futures

POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)

class MainWindow(QtWidgets.QMainWindow):
    glossReady = QtCore.Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live-OCR-Overlay")
        self.resize(1000, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.video = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.video, 3)

        # Horizontal layout for list and details
        hlayout = QtWidgets.QHBoxLayout()
        vbox.addLayout(hlayout, 2)

        self.list = QtWidgets.QListWidget()
        self.list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setStyleSheet("QListWidget{padding:4px;} QListWidget::item{margin:2px 0;}")
        hlayout.addWidget(self.list, 2)

        # Details view
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenExternalLinks(True)
        hlayout.addWidget(self.details, 3)

        self.words: Dict[str, Dict] = {}
        self.glosses: Dict[str, List[str]] = {}

        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=NUM_WORKERS)

        self.last_roi_count = 0
        self.fc = 0
        self.t = QtCore.QElapsedTimer()
        self.t.start()
        QtCore.QTimer.singleShot(1000, self.update_fps)

        self.glossReady.connect(self._update_gloss)

        # Start ScreenGrabber
        self.screen_grabber = ScreenGrabber()
        self.screen_grabber.frameCaptured.connect(self.on_frame)
        self.screen_grabber.start()

        # Loading label
        self.loading_label = QtWidgets.QLabel("Loading OCR workers, please wait...")
        self.loading_label.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(self.loading_label)
        self.list.setEnabled(False)
        self.details.setEnabled(False)

        QtCore.QTimer.singleShot(100, self.init_workers_async)

        # Prune timer
        QtCore.QTimer.singleShot(1000, self.prune_timer)

        self.list.itemSelectionChanged.connect(self.update_details)

        self.mouse_update_timer = QtCore.QTimer(self)
        self.mouse_update_timer.timeout.connect(self.refresh_list)
        self.mouse_update_timer.start(50)  # 20 times per second

    def init_workers_async(self):
        # Use a QThreadPool for async init
        from concurrent.futures import ThreadPoolExecutor

        def worker_init():
            self.workers = [OCRWorker(self.q) for _ in range(NUM_WORKERS)]
            for w in self.workers:
                w.setParent(self)
                w.linesFound.connect(self.on_lines)
                w.start()

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(worker_init)

        def on_done(_):
            self.loading_label.hide()
            self.list.setEnabled(True)
            self.details.setEnabled(True)

        # Use QTimer to poll for completion
        def check_future():
            if future.done():
                on_done(None)
            else:
                QtCore.QTimer.singleShot(100, check_future)
        check_future()

    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame):
        # Send frame to OCR every OCR_INTERVAL frames
        if self.fc % OCR_INTERVAL == 0:
            try:
                self.q.put_nowait(frame.copy())
            except queue.Full:
                pass

        h, w = frame.shape[:2]
        img = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format_BGR888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.fc += 1

    @QtCore.Slot(list)
    def on_lines(self, lines):
        for word, bbox in lines:
            meta = self.words.setdefault(word, {"freq": 0, "miss": 0, "bbox": bbox})
            meta["freq"] += 1
            meta["miss"] = 0
            meta["bbox"] = bbox
            self._submit_translate(word)
        self.refresh_list()

    def _submit_translate(self, word: str):
        if word in self.glosses:
            return

        def _worker():
            try:
                data = translate.translate(word.lower())
                return data  # Return the full translation data!
            except Exception:
                return {}

        def _done(fut):
            self.glossReady.emit(word, fut.result())

        POOL.submit(_worker).add_done_callback(_done)

    @QtCore.Slot(str, dict)
    def _update_gloss(self, word: str, translation_data: dict):
        if translation_data:
            self.glosses[word] = translation_data
            self.refresh_list()
            self.update_details()

    def refresh_list(self):
        # Save scroll position and selected word
        scroll_pos = self.list.verticalScrollBar().value()
        selected_items = self.list.selectedItems()
        selected_word = None
        if selected_items:
            selected_lbl = self.list.itemWidget(selected_items[0])
            if selected_lbl:
                selected_word = html.unescape(selected_lbl.text().split("<br>")[0].replace("<b>", "").replace("</b>", ""))

        # Compute mouse position in capture area coordinates
        mouse_video_pos = None
        global_mouse_pos = QtGui.QCursor.pos()
        mx, my = global_mouse_pos.x(), global_mouse_pos.y()

        # Use the monitor/capture area geometry
        capture_x = self.screen_grabber.capture_x
        capture_y = self.screen_grabber.capture_y
        capture_w = self.screen_grabber.capture_w
        capture_h = self.screen_grabber.capture_h

        if (capture_x <= mx < capture_x + capture_w) and (capture_y <= my < capture_y + capture_h):
            mouse_video_pos = (mx - capture_x, my - capture_y)
            # Mouse is over the captured monitor
        else:
            mouse_video_pos = None
            # Mouse is not over the captured monitor

        scored = []
        for word, meta in self.words.items():
            bbox = meta["bbox"]
            # print(f"[DEBUG] Word '{word}' bbox: {bbox}")
            # Compute distance from mouse to bbox center
            if mouse_video_pos is not None:
                x0, y0, x1, y1 = bbox
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                dist = ((mouse_video_pos[0] - cx) ** 2 + (mouse_video_pos[1] - cy) ** 2) ** 0.5
                # print(f"[DEBUG] Distance from mouse to '{word}': {dist:.1f}")
            else:
                dist = float('inf')
            score = meta["freq"]
            # Sort by distance (closer first), then by frequency (higher first)
            scored.append((dist, -score, word))

        scored.sort()
        self.list.clear()
        for _, _, word in scored:
            gloss_data = self.glosses.get(word, {})
            glosses = [s["definition"] for s in gloss_data.get("senses", [])][:3] if gloss_data else []
            if glosses:
                html_txt = f"<b>{html.escape(word)}</b><br><i>{html.escape(', '.join(glosses))}</i>"
            else:
                html_txt = f"<b>{html.escape(word)}</b>"

            item = QtWidgets.QListWidgetItem()
            lbl = QtWidgets.QLabel(html_txt)
            lbl.setTextFormat(QtCore.Qt.RichText)
            lbl.setWordWrap(True)
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

            item.setSizeHint(lbl.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, lbl)

            # Restore selection if this is the previously selected word
            if selected_word and html.unescape(word) == selected_word:
                item.setSelected(True)

        # Restore scroll position
        self.list.verticalScrollBar().setValue(scroll_pos)
        self.update_details()

    def update_fps(self):
        fps = self.fc / max(1, self.t.elapsed() / 1000)
        self.setWindowTitle(
            f"Live-OCR-Overlay  |  {fps:.0f} FPS  |  {len(self.words)} words  "
            f"|  ROI {self.last_roi_count}"
        )
        self.fc = 0
        self.t.restart()
        QtCore.QTimer.singleShot(1000, self.update_fps)

    def prune(self):
        for w in list(self.words):
            self.words[w]["miss"] += 1
            if self.words[w]["miss"] >= WORD_MAX_MISSES:
                del self.words[w]
        self.refresh_list()

    def prune_timer(self):
        # Prune words that have exceeded the miss limit
        self.prune()
        # Restart the timer
        QtCore.QTimer.singleShot(1000, self.prune_timer)

    def update_details(self):
        # If a word is selected, use that; otherwise, use the top word in the list
        selected_items = self.list.selectedItems()
        if selected_items:
            selected_lbl = self.list.itemWidget(selected_items[0])
        else:
            # No selection: use the first item in the list
            if self.list.count() == 0:
                self.details.clear()
                return
            first_item = self.list.item(0)
            selected_lbl = self.list.itemWidget(first_item)
        if not selected_lbl:
            self.details.clear()
            return
        word = html.unescape(selected_lbl.text().split("<br>")[0].replace("<b>", "").replace("</b>", ""))
        gloss_data = self.glosses.get(word, None)
        details_html = f"<h2>{html.escape(word)}</h2>"
        if gloss_data is None:
            details_html += "<i>Translating...</i>"
        elif gloss_data:
            senses = gloss_data.get("senses", [])
            if senses:
                details_html += "<ul>"
                for s in senses:
                    details_html += f"<li><b>{html.escape(s.get('definition', ''))}</b>"
                    # Add more details if available
                    if "examples" in s:
                        details_html += "<ul>" + "".join(f"<li>{html.escape(ex)}</li>" for ex in s["examples"]) + "</ul>"
                    details_html += "</li>"
                details_html += "</ul>"
            else:
                details_html += "<i>No senses found.</i>"
        else:
            details_html += "<i>No translation available.</i>"
        self.details.setHtml(details_html)
        # Do not scroll the selected word to the top!

    def closeEvent(self, event):
        # Stop screen grabber
        if hasattr(self, "screen_grabber"):
            self.screen_grabber.stop()
        # Stop OCR workers
        if hasattr(self, "workers"):
            for w in self.workers:
                w.stop()
        event.accept()
        self.update_details()