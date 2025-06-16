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
## Google Translation API
Requires a `.env` file with the following content:

`GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"`

## General Required Packages
* pipewire
* gstreamer
* gst-plugin-pipewire

Currently only supports Python 3.11

## Wayland Support
On gnome it should just work

On KDE, it requires a virtual output when choosing screenshare option