from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QMetaObject, Qt
import html
from typing import Dict, List
import numpy as np
from screen_capture.screen_grabber import ScreenGrabber
import concurrent.futures
import time

import sys, os
sys.path.append(os.path.dirname(__file__))  # Ensure src/ is in sys.path

from helpers.logging import debug

POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)
server = "http://localhost:8000"  # Replace with your server URL

dynamic_ocr_interval = 5.0  # seconds

def request_server_ocr(frame: np.ndarray) -> concurrent.futures.Future:
    """
    Submit an OCR request to the server
    """
    import requests
    from PIL import Image
    from io import BytesIO
    # sending to this endpoint
    """
    @app.post("/analyze/image")
    async def analyze_image_base64(payload: ImageRequest):
    """
    # Convert the frame to PIL Image
    img = Image.fromarray(frame)
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    # Encode to base64
    import base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    # Prepare the payload
    payload = {
        "image_base64": img_base64
    }
    # Send the request
    future = POOL.submit(
        requests.post,
        f"{server}/analyze/image",
        json=payload
    )
    return future


def request_server_translate(text: str) -> concurrent.futures.Future:
    """
    Submit a translation request to the server
    """
    import requests
    # Prepare the payload
    payload = {
        "text": text
    }
    # Send the request
    future = POOL.submit(
        requests.post,
        f"{server}/translate",
        json=payload
    )
    return future

def request_server_translate_batch(text_list: list[str]) -> concurrent.futures.Future:
    """
    Submit a translation request to the server
    """
    import requests
    # Prepare the payload
    payload = []
    for text in text_list:
        word = {
            "text": text
        }
        payload.append(word)

    # Send the request
    future = POOL.submit(
        requests.post,
        f"{server}/translate/batch",
        json=payload
    )
    return future

class MainWindow(QtWidgets.QMainWindow):
    glossReady = QtCore.Signal(str, dict)
    toggle_mouse_follow_mode = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR-Translate")
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
        self.search_bar.returnPressed.connect(request_server_translate)

        # --- Mouse Follow Mode Toggle ---
        self.mouse_follow_checkbox = QtWidgets.QCheckBox("Mouse Follow Mode")
        self.mouse_follow_checkbox.setChecked(False)
        self.mouse_follow_checkbox.stateChanged.connect(self.toggle_mouse_follow_mode)
        left_vbox.addWidget(self.mouse_follow_checkbox)

        self.mouse_follow_mode = False  # Default enabled

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
        # with the following structure:
        # {
        #     "word": {
        #         "bbox": (x0, y0, x1, y1),  # Bounding box coordinates
        #         "freq": int,  # Frequency of the word
        #         "last_seen": float,  # Last seen timestamp
        #         "translated": {
        #             "word": "guten morgen",
        #             "definitions": {
        #                 "ADJ(A)": "guten morg -> good morning"
        #             },
        #             "html": "<span class=\"translation\">good morning</span>"
        #         }
        #     }
        # }

        self.fc = 0
        self.t = QtCore.QElapsedTimer()
        self.t.start()
        QtCore.QTimer.singleShot(1000, self.update_fps)

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

        # Prune timer
        QtCore.QTimer.singleShot(1000, self.prune_timer)

        self.list.itemSelectionChanged.connect(self.update_details)

        self.mouse_update_timer = QtCore.QTimer(self)
        self.mouse_update_timer.timeout.connect(self.refresh_list)
        self.mouse_update_timer.start(150)  # 20 times per second

        self._search_active = False

        self.last_ocr_time = time.monotonic()

    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame):
        now = time.monotonic()
        global dynamic_ocr_interval
        if now - self.last_ocr_time >= dynamic_ocr_interval:
            print("[DEBUG] Pushing frame to server")
            future = request_server_ocr(frame)
            future.add_done_callback(self.ocr_on_future_done)
            self.last_ocr_time = now  # Update last OCR time immediately
            def update_last_ocr_time(_):
                self.last_ocr_time = now
            future.add_done_callback(update_last_ocr_time)
        self.fc += 1

    def ocr_on_future_done(self, future):
        try:
            result = future.result().json()
            if "error" in result:
                print("[ERROR] OCR request failed:", result["error"])
                return
            if "texts" not in result or not result["texts"]:
                print("[ERROR] No texts found in the OCR result.")
                return

            # --- ENABLE widgets after first OCR result ---
            if not self.list.isEnabled():
                QMetaObject.invokeMethod(self.list, "setEnabled", Qt.QueuedConnection, QtCore.Q_ARG(bool, True))
            if not self.details.isEnabled():
                QMetaObject.invokeMethod(self.details, "setEnabled", Qt.QueuedConnection, QtCore.Q_ARG(bool, True))
            if self.loading_label.isVisible():
                QMetaObject.invokeMethod(self.loading_label, "hide", Qt.QueuedConnection)

            words_to_translate = []
            for entry in result["texts"]:
                text  = entry["text"].strip()
                bbox  = entry["bbox"]
                if not text:
                    continue

                meta = self.words.setdefault(text, {"bbox": bbox, "freq": 0, "last_seen": 0})
                meta["bbox"]      = bbox            # always keep most-recent bbox
                meta["last_seen"] = time.monotonic()
                meta["freq"]     += 1

                words_to_translate.append(text)
            if words_to_translate:
                trans_future = request_server_translate_batch(words_to_translate)
                trans_future.add_done_callback(self.on_translate_future_done)

            # Safe refresh in GUI thread
            QMetaObject.invokeMethod(
                self, "refresh_list", Qt.QueuedConnection
            )
        except Exception as e:
            print("[ERROR] OCR failed:", e)

    def on_translate_future_done(self, future):
        """
        structure of response:
        "translated": {
            "word": "guten morgen",
            "definitions": {
                "ADJ(A)": "guten morg -> good morning"
            },
            "html": "<span class=\"translation\">good morning</span>"
        }
        """
        try:
            result = future.result().json()
            # check if list, if not, make to list of 1
            if not result:
                print("[ERROR] No translations found in the result.")
                return
            for word_data in result['translated']:
                # Update the words dictionary with translation
                for word, meta in self.words.items():
                    if word == word_data["word"]:
                        meta["translated"] = word_data
                        break
            # Safe refresh in GUI thread
            QMetaObject.invokeMethod(
                self, "refresh_list", Qt.QueuedConnection
            )
        except Exception as e:
            print("[ERROR] Translation failed:", e.with_traceback())
            # print more debug

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
            # Compute distance from mouse to bbox center
            bbox = meta["bbox"]
            if mouse_video_pos is not None:
                x0, y0, x1, y1 = bbox
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                dist = ((mouse_video_pos[0] - cx) ** 2 + (mouse_video_pos[1] - cy) ** 2) ** 0.5
            else:
                dist = float('inf')
            score = meta["freq"]
            scored.append((dist, -score, word))

        scored.sort()
        self.list.clear()
        MAX_LIST = 40  # Only show top 40 words
        for _, _, word in scored[:MAX_LIST]:
            gloss_data = self.words[word].get("translated", {})
            definitions = gloss_data.get("definitions", {})
            first_def = ""
            first_pos = ""
            for pos, defs in definitions.items():
                first_def = defs
                first_pos = pos
                break
            if first_def:
                # --- Make definition clickable if possible ---
                html_txt = f"<b>{html.escape(word)}</b><br><a href=\"#def\">{html.escape(first_def)}</a>"
            else:
                html_txt = f"<b>{html.escape(word)}</b>"

            item = QtWidgets.QListWidgetItem()
            lbl = QtWidgets.QLabel(html_txt)
            lbl.setTextFormat(QtCore.Qt.RichText)
            lbl.setWordWrap(True)
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            lbl.setOpenExternalLinks(False)
            lbl.linkActivated.connect(lambda link, w=word: self.show_definition(w))

            item.setSizeHint(lbl.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, lbl)

            if selected_word and html.unescape(word) == selected_word:
                item.setSelected(True)

        self.list.verticalScrollBar().setValue(scroll_pos)
        if not self.search_bar.text().strip():
            self.update_details()

    def show_definition(self, word):
        # Show full details for the word when link is clicked
        self.details.setHtml(self.render_details_html(word))

    def render_details_html(self, word):
        # Check self.words[word] exists
        if word not in self.words:
            return f"<h2>{html.escape(word)}</h2><i>Word not found</i>"
        gloss_data = self.words[word].get("translated", {})
        details_html = f"<h2>{html.escape(word)}</h2>"
        if not gloss_data:
            details_html += "<i>Translating...</i>"
        else:
            html_content = gloss_data.get("html", "")
            details_html += html_content
        return details_html

    def update_fps(self):
        fps = self.fc / max(1, self.t.elapsed() / 1000)
        self.setWindowTitle(
            f"Live-OCR-Overlay  |  {fps:.0f} FPS  |  {len(self.words)} words  |  {dynamic_ocr_interval/2:.2f}s last process"
        )
        self.fc = 0
        self.t.restart()
        QtCore.QTimer.singleShot(1000, self.update_fps)

    def prune(self):
        now = time.monotonic()
        for w in list(self.words):
            if now - self.words[w]["last_seen"] >= 60:
                del self.words[w]
        self.refresh_list()

    def prune_timer(self):
        self.prune()
        QtCore.QTimer.singleShot(1000, self.prune_timer)

    def update_details(self):
        # Only update details if search bar is empty
        if self.search_bar.text().strip():
            return
        self._search_active = False
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
        self.details.setHtml(self.render_details_html(word))