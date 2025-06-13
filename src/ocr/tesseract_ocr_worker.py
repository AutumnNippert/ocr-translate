from PySide6 import QtCore
import pytesseract
import numpy as np
from time import time
import cv2

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Adjust if needed

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
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Optional: adaptive thresholding for better contrast
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                              cv2.THRESH_BINARY, 11, 2)
        data = pytesseract.image_to_data(
            gray, lang='deu', config="--oem 1 --psm 11", output_type=pytesseract.Output.DICT
        )
        words = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf_val = data['conf'][i]
            try:
                conf = int(conf_val)
            except Exception:
                conf = 0
            if conf > 30 and len(text) > 1:
                x0, y0, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                x1, y1 = x0 + w, y0 + h
                for word in clean_text(text):
                    if word:
                        words.append((word, (x0, y0, x1, y1)))
        words = list(set(words))
        words = [(word, bbox) for word, bbox in words if word]
        if words:
            self.linesFound.emit(words)
        currtime = time() - currtime
        print(f"Tesseract OCR processed {len(words)} words in {currtime * 1000:.2f} ms")