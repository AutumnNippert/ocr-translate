# Screen Capture Translate

## Overview
Screen Capture Translate is a live OCR overlay application that captures a region of your screen, recognizes words using OCR, and displays real-time translations in a user-friendly interface.

## Features
- Live screen capture from any monitor
- Real-time OCR using Tesseract or EasyOCR
- Instant translation of recognized words (German → English)
- Responsive UI with word list prioritized by mouse position
- Details panel with definitions and example sentences

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd screen-cap-translate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   - On Ubuntu: `sudo apt install tesseract-ocr`
   - On Windows: [Download from UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

Run the application with:
```bash
python src/main.py
```

## Project Structure

```
screen-cap-translate/
├── src/
│   ├── main.py
│   ├── constants.py
│   ├── ocr/
│   │   ├── screen_grabber.py
│   │   ├── tesseract_ocr_worker.py
│   │   └── easyocr_worker.py
│   ├── translation/
│   │   └── translate.py
│   └── ui/
│       └── main_window.py
├── requirements.txt
└── README.md
```

## Dependencies

- PySide6
- numpy
- opencv-python
- pytesseract
- mss
- requests
- easyocr
- wiktionaryparser
- PyMultiDictionary
- word2word

See `requirements.txt` for exact versions.

## Notes

- Tesseract OCR must be installed and available in your system path.
- For best results, use a GPU for EasyOCR (optional).
- The application is designed for German-to-English translation but can be extended.

## License

MIT License