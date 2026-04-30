# config.py — Global Configuration & Constants
# Capture settings, model paths, confidence thresholds, color maps, queue sizes.

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# Model paths
MODEL_PATH = os.path.join("models", "yolov8n.pt")

# -----------------------------------------------------------------------------
# Capture Settings
# -----------------------------------------------------------------------------

# Monitor to capture (0 = primary, 1 = secondary, etc.)
MONITOR_INDEX = 0

# Capture resolution (None = full monitor resolution)
# Common resolutions: (1920, 1080), (2560, 1440), (3840, 2160)
CAPTURE_RESOLUTION = None

# FPS limit for capture (None = unlimited)
CAPTURE_FPS = 60

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------

# Confidence threshold (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.45

# IOU threshold for Non-Maximum Suppression
IOU_THRESHOLD = 0.5

# Model input size (pixels) - smaller = faster, larger = more accurate
MODEL_INPUT_SIZE = 416  # Reduced from 640 for better performance

# -----------------------------------------------------------------------------
# Performance Settings
# -----------------------------------------------------------------------------

# Inference skip rate - run inference on every Nth frame (1 = every frame, 2 = every other frame)
# Higher values = better performance but less responsive detection
INFERENCE_SKIP_RATE = 2  # Run inference on every other frame

# Queue sizes - increased to prevent dropping frames during heavy load
CAPTURE_QUEUE_SIZE = 4
DETECTION_QUEUE_SIZE = 4

# -----------------------------------------------------------------------------
# Device Settings
# -----------------------------------------------------------------------------

# Use GPU if available, otherwise CPU
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# -----------------------------------------------------------------------------
# Color Map for Bounding Boxes (R, G, B for PyQt5)
# -----------------------------------------------------------------------------

# Class names and colors
CLASS_NAMES = {
    0: "CT",
    1: "T"
}

COLOR_MAP = {
    0: (255, 50, 50),     # CT: Red
    1: (50, 255, 50),     # T: Green
}

# Default color if class is not found
DEFAULT_COLOR = (255, 255, 255)  # White

# Bounding box line thickness
BOX_THICKNESS = 2

# Font size for labels
FONT_SIZE = 14

# -----------------------------------------------------------------------------
# Overlay Settings
# -----------------------------------------------------------------------------

# Always on top
ALWAYS_ON_TOP = True

# Click-through (transparent to mouse events)
CLICK_THROUGH = True

# Overlay refresh rate (ms)
OVERLAY_REFRESH_MS = 16  # ~60 FPS

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def get_device():
    """Return the device to use for inference."""
    return DEVICE

def get_model_path():
    """Return the path to the YOLO model."""
    return MODEL_PATH

def get_capture_resolution():
    """Return the capture resolution tuple (width, height)."""
    return CAPTURE_RESOLUTION

def get_confidence_threshold():
    """Return the confidence threshold for detection."""
    return CONFIDENCE_THRESHOLD

def get_color(class_id):
    """Return the color (R, G, B) for a specific class ID."""
    return COLOR_MAP.get(class_id, DEFAULT_COLOR)

def get_class_name(class_id):
    """Return the human-readable class name for a class ID."""
    return CLASS_NAMES.get(class_id, f"Class {class_id}")

def get_monitor_index():
    """Return the monitor index to capture."""
    return MONITOR_INDEX
