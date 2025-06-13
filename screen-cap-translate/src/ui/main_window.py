from PySide6 import QtCore, QtGui, QtWidgets
import html
import queue
from typing import Dict, List

class MainWindow(QtWidgets.QMainWindow):
    glossReady = QtCore.Signal(str, list)

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

        self.list = QtWidgets.QListWidget()
        self.list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setStyleSheet("QListWidget{padding:4px;} QListWidget::item{margin:2px 0;}")
        vbox.addWidget(self.list, 2)

        self.words: Dict[str, Dict] = {}
        self.glosses: Dict[str, List[str]] = {}

        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)

        self.last_roi_count = 0
        self.fc = 0
        self.t = QtCore.QElapsedTimer()
        self.t.start()
        QtCore.QTimer.singleShot(1000, self.update_fps)

        self.glossReady.connect(self._update_gloss)

    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame):
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

    @QtCore.Slot(str, list)
    def _update_gloss(self, word: str, gloss_list: list):
        if gloss_list:
            self.glosses[word] = gloss_list[:3]
            self.refresh_list()

    def _submit_translate(self, word: str):
        if word in self.glosses:
            return

        def _worker():
            try:
                data = translate.translate(word.lower())
                return [s["definition"] for s in data.get("senses", [])][:3]
            except Exception:
                return []

        def _done(fut):
            self.glossReady.emit(word, fut.result())

        POOL.submit(_worker).add_done_callback(_done)

    def refresh_list(self):
        scored = []
        for word, meta in self.words.items():
            score = meta["freq"]
            scored.append((score, word))
        scored.sort(reverse=True)
        self.list.clear()
        for _, word in scored:
            gloss = ", ".join(self.glosses.get(word, []))
            if gloss:
                html_txt = f"<b>{html.escape(word)}</b><br><i>{html.escape(gloss)}</i>"
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

    def update_fps(self):
        fps = self.fc / max(1, self.t.elapsed() / 1000)
        self.setWindowTitle(
            f"Live-OCR-Overlay  |  {fps:.0f} FPS  |  {len(self.words)} words  "
            f"|  ROI {self.last_roi_count}"
        )
        self.fc = 0
        self.t.restart()
        QtCore.QTimer.singleShot(1000, self.update_fps)

    def closeEvent(self, e):
        super().closeEvent(e)