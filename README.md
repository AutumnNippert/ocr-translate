# [Whatever name this becomes later]

## Overview
This is a live OCR application that captures a region of your screen, recognizes words using OCR, and displays real-time word translations.

## Features
- Live screen capture from any monitor (ish)
- Real-time OCR using paddleocr
- Instant translation of recognized words (German → English)
- Responsive UI with word list prioritized by mouse position (X11 only)
- Details panel with detailed breakdown of word information

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

## Usage

Run the application with:
```bash
python src/main.py
```

# Other
## Wayland Support
On gnome it should just work

On KDE, it requires a virtual output when choosing screenshare option

### REQUIRED PACKAGES:
* pipewire
* gstreamer
* gst-plugin-pipewire