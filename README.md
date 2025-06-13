# Screen Capture Translate

## Overview
Screen Capture Translate is a live OCR overlay application that captures a specified region of the screen, processes the captured images to extract text using Optical Character Recognition (OCR), and displays the recognized words along with their translations in real-time.

## Features
- Fast region of interest (ROI) detection for efficient text extraction.
- Deep OCR processing for improved accuracy on small fonts.
- Real-time translation of recognized words.
- User-friendly interface for displaying captured video and recognized text.

## Project Structure
```
screen-cap-translate
├── src
│   ├── main.py               # Entry point of the application
│   ├── ocr                   # OCR module
│   │   ├── __init__.py       # Initialization file for OCR module
│   │   └── ocr_worker.py      # Contains OCRWorker class for processing frames
│   ├── ui                    # UI module
│   │   ├── __init__.py       # Initialization file for UI module
│   │   └── main_window.py     # Contains MainWindow class for user interface
│   └── utils                 # Utility functions
│       ├── __init__.py       # Initialization file for utils module
│       └── helpers.py         # Contains helper functions for image processing
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd screen-cap-translate
pip install -r requirements.txt
```

## Usage
To run the application, execute the following command:

```bash
python src/main.py
```

## Dependencies
This project requires the following Python packages:
- PySide6
- pytesseract
- numpy
- opencv-python
- mss

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.