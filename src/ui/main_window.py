from PySide6 import QtCore, QtGui, QtWidgets
import html
import queue
from typing import Dict, List
import numpy as np
from translation.wiktionary_translate import translate
from ocr.screen_grabber import ScreenGrabber
from ocr.paddle_ocr_worker import OCRWorker
import concurrent.futures
import time
import string 

dynamic_ocr_interval = 1

import sys, os
sys.path.append(os.path.dirname(__file__))  # Ensure src/ is in sys.path

from helpers.logging import debug

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

        # Horizontal layout for list and details
        hlayout = QtWidgets.QHBoxLayout()
        vbox.addLayout(hlayout, 2)

        # --- LEFT PANEL: search bar above list ---
        left_vbox = QtWidgets.QVBoxLayout()
        hlayout.addLayout(left_vbox, 2)

        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Search for a word...")
        left_vbox.addWidget(self.search_bar)
        self.search_bar.returnPressed.connect(self.translate_search_word)

        # --- Mouse Follow Mode Toggle ---
        self.mouse_follow_checkbox = QtWidgets.QCheckBox("Mouse Follow Mode")
        self.mouse_follow_checkbox.setChecked(True)
        self.mouse_follow_checkbox.stateChanged.connect(self.toggle_mouse_follow_mode)
        left_vbox.addWidget(self.mouse_follow_checkbox)

        self.mouse_follow_mode = True  # Default enabled

        self.list = QtWidgets.QListWidget()
        self.list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setStyleSheet("QListWidget{padding:4px;} QListWidget::item{margin:2px 0;}")
        left_vbox.addWidget(self.list)

        # --- RIGHT PANEL: details ---
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenExternalLinks(True)
        hlayout.addWidget(self.details, 3)

        self.words: Dict[str, Dict] = {}
        self.glosses: Dict[str, List[str]] = {}

        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)

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
        self.mouse_update_timer.start(150)  # 20 times per second

        self._search_active = False

        self.last_ocr_time = time.monotonic()

    def init_workers_async(self):
        # Use a QThreadPool for async init
        from concurrent.futures import ThreadPoolExecutor

        def worker_init():
            self.workers = [OCRWorker(self.q) for _ in range(1)]
            for w in self.workers:
                w.setParent(self)
                w.linesFound.connect(self.on_lines)
                w.process_time.connect(self.on_ocr_process_complete)
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
        now = time.monotonic()
        global dynamic_ocr_interval
        if now - self.last_ocr_time >= dynamic_ocr_interval:
            try:
                print("[DEBUG] Pushing frame to OCR queue")
                self.q.put_nowait(frame.copy())
            except queue.Full:
                pass
            self.last_ocr_time = now

        # h, w = frame.shape[:2]
        # img = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format_BGR888)
        # self.video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
        #     self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.fc += 1

    @QtCore.Slot(list)
    def on_lines(self, lines):
        t0 = time.time()
        now = time.monotonic()
        for word, bbox in lines:
            word_clean = word.translate(str.maketrans('', '', string.punctuation))
            if not word_clean.strip():
                continue  # Skip empty words after cleaning
            meta = self.words.setdefault(word_clean, {"freq": 0, "last_seen": now, "bbox": bbox})
            meta["freq"] += 1
            meta["last_seen"] = now
            meta["bbox"] = bbox
        t1 = time.time()
        self.refresh_list()
        t2 = time.time()
        self.submit_priority_translations(n=5)
        t3 = time.time()
        # print(f"[DEBUG] on_lines: update words {1000*(t1-t0):.1f}ms, refresh_list {1000*(t2-t1):.1f}ms, submit_priority_translations {1000*(t3-t2):.1f}ms")

    @QtCore.Slot(float)
    def on_ocr_process_complete(self, time):
        global dynamic_ocr_interval
        print("[SUPERDEBUGTHISISADEBUGMESSAGEHI]Setting dynamic OCR interval to", time * 5)
        dynamic_ocr_interval = time

    def toggle_mouse_follow_mode(self, state):
        self.mouse_follow_mode = bool(state)
        self.refresh_list()

    def refresh_list(self):
        t0 = time.time()
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
        if self.mouse_follow_mode:
            global_mouse_pos = QtGui.QCursor.pos()
            mx, my = global_mouse_pos.x(), global_mouse_pos.y()
            capture_x = self.screen_grabber.capture_x
            capture_y = self.screen_grabber.capture_y
            capture_w = self.screen_grabber.capture_w
            capture_h = self.screen_grabber.capture_h

            if (capture_x <= mx < capture_x + capture_w) and (capture_y <= my < capture_y + capture_h):
                mouse_video_pos = (mx - capture_x, my - capture_y)
            else:
                mouse_video_pos = None
        else:
            mouse_video_pos = None

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
        t1 = time.time()
        self.list.clear()
        MAX_LIST = 20  # Only show top 20 words
        for _, _, word in scored[:MAX_LIST]:
            gloss_data = self.glosses.get(word, {})
            # --- Use ['definitions'] for the list ---
            definitions = gloss_data.get("definitions", {})
            first_def = ""
            # Get the first available definition from any part of speech
            for pos_defs in definitions.values():
                if isinstance(pos_defs, list) and pos_defs:
                    first_def = pos_defs[0].get("definition", "")
                    break
            if first_def:
                html_txt = f"<b>{html.escape(word)}</b><br><i>{html.escape(first_def)}</i>"
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
        # Only update details if search bar is empty
        if not self.search_bar.text().strip():
            self.update_details()
        t2 = time.time()
        # print(f"[DEBUG] refresh_list: scoring {1000*(t1-t0):.1f}ms, list update {1000*(t2-t1):.1f}ms, total {1000*(t2-t0):.1f}ms")

    def update_fps(self):
        fps = self.fc / max(1, self.t.elapsed() / 1000)
        self.setWindowTitle(
            f"Live-OCR-Overlay  |  {fps:.0f} FPS  |  {len(self.words)} words  |  {dynamic_ocr_interval}s last process"
        )
        self.fc = 0
        self.t.restart()
        QtCore.QTimer.singleShot(1000, self.update_fps)

    def prune(self):
        now = time.monotonic()
        for w in list(self.words):
            if now - self.words[w]["last_seen"] >= dynamic_ocr_interval*2:
                del self.words[w]
        self.refresh_list()

    def prune_timer(self):
        # Prune words that have exceeded the miss limit
        self.prune()
        # Restart the timer
        QtCore.QTimer.singleShot(1000, self.prune_timer)

    def update_details(self):
        # Only update details if search bar is empty
        if self.search_bar.text().strip():
            return
        self._search_active = False  # <-- Add this line at the start
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
        # --- Render the 'html' field for details ---
        if gloss_data is None:
            details_html += "<i>Translating...</i>"
        else:
            html_content = gloss_data.get("html", "")
            details_html += html_content
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

    def get_top_words(self, n=3):
        if self.mouse_follow_mode:
            global_mouse_pos = QtGui.QCursor.pos()
            mx, my = global_mouse_pos.x(), global_mouse_pos.y()
            capture_x = self.screen_grabber.capture_x
            capture_y = self.screen_grabber.capture_y
            capture_w = self.screen_grabber.capture_w
            capture_h = self.screen_grabber.capture_h

            if (capture_x <= mx < capture_x + capture_w) and (capture_y <= my < capture_y + capture_h):
                mouse_video_pos = (mx - capture_x, my - capture_y)
            else:
                mouse_video_pos = None
        else:
            mouse_video_pos = None

        scored = []
        for word, meta in self.words.items():
            bbox = meta["bbox"]
            if mouse_video_pos is not None:
                x0, y0, x1, y1 = bbox
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                dist = ((mouse_video_pos[0] - cx) ** 2 + (mouse_video_pos[1] - cy) ** 2) ** 0.5
            else:
                dist = float('inf')
            scored.append((dist, word))

        scored.sort()
        result = []
        for _, word in scored[:n]:
            gloss_data = self.glosses.get(word, {})
            glosses = [s["definition"] for s in gloss_data.get("senses", [])][:2] if gloss_data else []
            gloss = ", ".join(glosses) if glosses else ""
            result.append((word, gloss))
        return result

    def submit_priority_translations(self, n=5):
        if self.mouse_follow_mode:
            global_mouse_pos = QtGui.QCursor.pos()
            mx, my = global_mouse_pos.x(), global_mouse_pos.y()
            capture_x = self.screen_grabber.capture_x
            capture_y = self.screen_grabber.capture_y
            capture_w = self.screen_grabber.capture_w
            capture_h = self.screen_grabber.capture_h

            if (capture_x <= mx < capture_x + capture_w) and (capture_y <= my < capture_y + capture_h):
                mouse_video_pos = (mx - capture_x, my - capture_y)
            else:
                mouse_video_pos = None
        else:
            mouse_video_pos = None

        scored = []
        for word, meta in self.words.items():
            if word in self.glosses:
                continue
            bbox = meta["bbox"]
            if mouse_video_pos is not None:
                x0, y0, x1, y1 = bbox
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                dist = ((mouse_video_pos[0] - cx) ** 2 + (mouse_video_pos[1] - cy) ** 2) ** 0.5
            else:
                dist = float('inf')
            scored.append((dist, word))
        scored.sort()
        print(f"[DEBUG] Submitting priority translations for {len(scored)} words")
        # prpint the whole list of words
        print(f"[DEBUG] Top {n} words to translate:")
        for i, (dist, word) in enumerate(scored[:n]):
            print(f"[DEBUG] {i+1}. '{word}' at distance {dist:.1f}")
        for _, word in scored[:n]:
            print(f"[DEBUG] Submitting priority translation for '{word}'")
            self._submit_translate(word)

    # --- Add this slot to safely update details from any thread ---
    @QtCore.Slot(str)
    def set_details_html(self, html_str):
        self.details.setHtml(html_str)

    def _submit_translate(self, word: str):
        t0 = time.time()
        if word in self.glosses:
            return

        def _worker():
            try:
                return translate(word)
            except Exception:
                return None

        def _done(fut):
            self.glossReady.emit(word, fut.result())
            t1 = time.time()
            print(f"[DEBUG] Translation for '{word}' took {1000*(t1-t0):.1f}ms")

        POOL.submit(_worker).add_done_callback(_done)

    @QtCore.Slot(str, dict)
    def _update_gloss(self, word: str, translation_data: dict):
        if translation_data:
            self.glosses[word] = translation_data
            self.refresh_list()
            self.update_details()

    def translate_search_word(self):
        word = self.search_bar.text().strip()
        if not word:
            self.refresh_list()
            return
        self.list.clearSelection()
        def _worker():
            try:
                return translate(word)
            except Exception:
                return None

        def _done(fut):
            result = fut.result()
            details_html = f"<h2>{html.escape(word)}</h2>"
            if not result:
                details_html += "<i>No translation found.</i>"
            else:
                html_content = result.get("html", "")
                details_html += html_content
            QtCore.QMetaObject.invokeMethod(self, "set_details_html", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, details_html))

        POOL.submit(_worker).add_done_callback(_done)
