# main.py — Application Entry Point
# Wires together ScreenGrabber, InferenceEngine, and OverlayWindow.

import sys
import signal
import queue

# Import torch BEFORE PyQt5 to avoid DLL initialization conflicts on Windows
# See: https://github.com/pytorch/pytorch/issues/166628
import torch

from PyQt5.QtWidgets import QApplication

import config
from capture.screen_grabber import ScreenGrabber
from detection.inference_engine import InferenceEngine
from overlay.overlay_window import OverlayWindow
from utils.fps_counter import FPSCounter


def main():
    """Launch the CS2 Enemy Recon pipeline."""

    print("=" * 50)
    print("  CS2 Enemy Recon — AI Vision Prototype")
    print("=" * 50)
    print(f"  Device       : {config.DEVICE}")
    print(f"  Model        : {config.MODEL_PATH}")
    print(f"  Confidence   : {config.CONFIDENCE_THRESHOLD}")
    print(f"  Monitor      : {config.MONITOR_INDEX}")
    print("=" * 50)

    # --- Create shared queues ---
    frame_queue = queue.Queue(maxsize=config.CAPTURE_QUEUE_SIZE)
    results_queue = queue.Queue(maxsize=config.DETECTION_QUEUE_SIZE)

    # --- Instantiate modules ---
    grabber = ScreenGrabber(frame_queue)
    engine = InferenceEngine(frame_queue, results_queue)

    # --- Start producer & worker threads ---
    print("[Main] Starting Screen Grabber...")
    grabber.start()

    print("[Main] Starting Inference Engine...")
    engine.start()

    # --- Start the PyQt5 overlay on the main thread ---
    print("[Main] Launching Overlay...")
    app = QApplication(sys.argv)

    # Allow Ctrl+C to close the app gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    overlay = OverlayWindow(results_queue)
    overlay.show()

    print("[Main] Pipeline running. Press Ctrl+C to exit.")

    exit_code = app.exec_()

    # --- Shutdown ---
    print("\n[Main] Shutting down...")
    grabber.stop()
    engine.stop()
    print("[Main] Done.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
