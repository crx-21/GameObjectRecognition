# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a real-time computer vision application for detecting player models in Counter-Strike 2 using YOLOv8. The application follows a **producer-worker-consumer** pipeline pattern:

```
[ScreenGrabber] --frame_queue--> [InferenceEngine] --results_queue--> [OverlayWindow]
     (Producer)                    (Worker)                    (Consumer)
```

### Modules

| Module | Role | File |
|--------|------|------|
| Screen Grabber | Producer — captures frames via DXcam | `capture/screen_grabber.py` |
| Inference Engine | Worker — runs YOLOv8 | `detection/inference_engine.py` |
| Overlay Window | Consumer — draws bounding boxes | `overlay/overlay_window.py` |
| Config | Global settings | `config.py` |
| FPS Counter | Utility — tracks frame rate | `utils/fps_counter.py` |

### Thread Model

- **ScreenGrabber**: Runs on a daemon thread, pushes frames to `frame_queue`
- **InferenceEngine**: Runs on a daemon thread, pulls frames, runs inference, pushes results
- **OverlayWindow**: Runs on Qt main thread, reads from `results_queue` via timer

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Run a single test (example)
python -m pytest tests/ -v  # if tests exist
```

## Configuration

All settings are in `config.py`:

| Setting | Key | Description |
|---------|-----|-------------|
| Model path | `MODEL_PATH` | YOLOv8 model file |
| Device | `DEVICE` | "cuda" or "cpu" |
| Confidence threshold | `CONFIDENCE_THRESHOLD` | 0.0–1.0 |
| Monitor index | `MONITOR_INDEX` | Which monitor to capture |
| FPS limit | `CAPTURE_FPS` | Frame rate for capture |

### Class Detection

| Class ID | Name | Color |
|----------|------|-------|
| 0 | CT | Red (255, 50, 50) |
| 1 | T | Green (50, 255, 50) |

## Development Notes

- DXcam is used for screen capture (Windows-only)
- PyQt5 provides the transparent, click-through overlay
- Queues have small buffers (size=2) to avoid memory buildup
- The application uses a frameless, always-on-top window with transparency

## Data Flow

1. **Capture**: DXcam captures game frames and puts them in `frame_queue`
2. **Inference**: YOLOv8 model processes each frame, outputs detection results
3. **Overlay**: PyQt5 window reads latest detections and draws colored boxes with labels

## Common Issues

- **DXcam capture fails**: Ensure game is in fullscreen or borderless window mode
- **No detections**: Check `CONFIDENCE_THRESHOLD` in config.py
- **Overlay not visible**: Verify monitor index matches your setup
- **GPU not used**: Check `torch.cuda.is_available()` returns True
