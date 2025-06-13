from PySide6 import QtCore
import easyocr
import numpy as np
from time import time
import cv2

reader = easyocr.Reader(['de'], gpu=True)  # Set gpu=True if you want to use GPU

def clean_text(text: str) -> list[str]:
    """
    Cleans the text by removing unwanted characters and normalizing it.
    """
    text = text.replace("'", "").replace('"', '').replace('`', '')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    return text.split()

class OCRWorker(QtCore.QThread):
    linesFound = QtCore.Signal(list)

    def __init__(self, q):
        super().__init__()
        self.q = q
        self._run = True

    def run(self):
        while self._run:
            try:
                frame = self.q.get(timeout=0.05)
            except Exception:
                continue
            self.process(frame)
            self.q.task_done()

    def stop(self):
        self._run = False
        self.wait()

    def process(self, frame: np.ndarray):
        currtime = time()
        # small = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        results = reader.readtext(frame)
        words = []
        for bbox, text, conf in results:
            if conf > 0.3 and len(text.strip()) > 1:
                x0 = int(min([pt[0] for pt in bbox]))
                y0 = int(min([pt[1] for pt in bbox]))
                x1 = int(max([pt[0] for pt in bbox]))
                y1 = int(max([pt[1] for pt in bbox]))
                text = clean_text(text)
                # words.append((text.strip(), (x0, y0, x1, y1)))
                for word in text:
                    if word:
                        words.append((word.strip(), (x0, y0, x1, y1)))
        words = list(set(words))
        # Remove empty words and duplicates
        words = [(word, bbox) for word, bbox in words if word]
        if words:
            self.linesFound.emit(words)
        currtime = time() - currtime
        print(f"OCR processed {len(words)} words in {currtime * 1000:.2f} ms")