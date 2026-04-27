# CS2 Enemy Recon — AI Vision Prototype

Real-time Computer Vision prototype for detecting player models in Counter-Strike 2.

## Features

- Real-time screen capture via DXcam
- YOLOv8-based player detection (Enemy / Friendly)
- Transparent click-through overlay with bounding boxes and confidence scores

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Architecture

| Module | Role | File |
|---|---|---|
| Screen Grabber | Producer — captures frames | `capture/screen_grabber.py` |
| Inference Engine | Worker — runs YOLOv8 | `detection/inference_engine.py` |
| Overlay | Consumer — draws bounding boxes | `overlay/overlay_window.py` |
