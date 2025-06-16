from PySide6 import QtCore
import numpy as np
from time import time
from paddleocr import PaddleOCR
from PIL import Image

import os
os.nice(10)  # lower priority

# Initialize PaddleOCR (lighter config for real-time)
ocr = PaddleOCR(
    ocr_version='PP-OCRv3',
    use_doc_orientation_classify=False,
    det_limit_side_len=512,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def clean_text(text: str) -> list[str]:
    text = text.replace("'", "").replace('"', '').replace('`', '')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    return text.split()

class OCRWorker(QtCore.QThread):
    linesFound = QtCore.Signal(list)
    process_time = QtCore.Signal(float)

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
        # convert frame to black and white png
        img = Image.fromarray(frame)
        print("[OCRWorker] Starting OCR process")
        results = ocr.predict(frame)[0]
        # print(f"[OCRWorker] PaddleOCR returned {results}")
        # write json results to file
        # results.save_to_json('paddle_ocr_results.json')
        # result is dict where ['rec_texts'] contains an array of strings (like whole sentences).
        # ['rec_boxes'] contains an array of bounding boxes for each string.
        # I would like to interpolate the bounding boxes to words based on how many words are in the string.
        if not results or not results['rec_texts']:
            print("[OCRWorker] No text found")
            return
        # Initialize lists to hold words and their bounding boxes
        print("[OCRWorker] Processing results")
        word_strings = []
        sentence_sizes = []
        bounding_boxes = []

        #get bounding boxes first
        for box in results['rec_boxes']:
            bounding_boxes.append((box[0], box[1], box[2], box[3]))

        for text in results['rec_texts']:
            # Clean the text and split into words
            cleaned_words = clean_text(text)
            if not cleaned_words:
                continue
            for word in cleaned_words:
                word_strings.append(word)
                print("[OCRWorker] Found word:", word)
            sentence_sizes.append(len(cleaned_words))

        word_bounding_boxes = []
        # Now interpolate bounding boxes to words
        # for each sentence, distribute the bounding box over the words
        for i, size in enumerate(sentence_sizes):
            if size == 0:
                continue
            box = bounding_boxes[i]
            x0, y0, x1, y1 = box
            # Calculate the width of each word's bounding box
            word_width = (x1 - x0) / size
            for j in range(size):
                word_x0 = int(x0 + j * word_width)
                word_x1 = int(word_x0 + word_width)
                word_bounding_boxes.append((word_strings.pop(0), (word_x0, y0, word_x1, y1)))
        

        if word_bounding_boxes:
            self.linesFound.emit(word_bounding_boxes)
        currtime = time() - currtime
        self.process_time.emit(currtime)
        print(f"OCR processed {len(word_bounding_boxes)} words in {currtime * 1000:.2f} ms")
