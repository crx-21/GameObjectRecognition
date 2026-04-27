# screen_grabber.py — Module 1: Screen Grabber (Producer)
# DXcam-based screen capture running on a daemon thread.
# Pushes frames into a thread-safe queue.

import threading
import queue
import time

import dxcam
import numpy as np

import config
from utils.fps_counter import FPSCounter


class ScreenGrabber:
    """
    Producer module — captures frames from the screen using DXcam
    and pushes them into a thread-safe queue for the inference engine.
    """

    def __init__(self, frame_queue: queue.Queue):
        """
        Args:
            frame_queue: Thread-safe queue to push captured frames into.
        """
        self._frame_queue = frame_queue
        self._running = False
        self._thread: threading.Thread | None = None
        self._camera: dxcam.DXCamera | None = None
        self._fps_counter = FPSCounter()
        self._last_fps_print = time.time()

    def start(self):
        """Initialize DXcam and start the capture thread."""
        self._running = True
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread = thread
        thread.start()

    def stop(self):
        """Signal the capture thread to stop and wait for it to finish."""
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _capture_loop(self):
        """Main capture loop — runs on a daemon thread."""
        try:
            # Create the DXcam camera instance
            self._camera = dxcam.create(
                device_idx=0,
                output_idx=config.MONITOR_INDEX,
                output_color="BGR",
            )

            # Determine the capture region
            region = None  # None = full screen
            if config.CAPTURE_RESOLUTION is not None:
                w, h = config.CAPTURE_RESOLUTION
                region = (0, 0, w, h)

            # Start DXcam's internal capture loop
            self._camera.start(
                region=region,
                target_fps=config.CAPTURE_FPS,
            )

            while self._running:
                frame = self._camera.get_latest_frame()

                if frame is not None:
                    # Update FPS counter
                    self._fps_counter.tick()

                    # Print FPS every 5 seconds
                    now = time.time()
                    if now - self._last_fps_print >= 5.0:
                        fps = self._fps_counter.fps
                        print(f"[ScreenGrabber] Capture FPS: {fps:.1f}")
                        self._last_fps_print = now

                    # If the queue is full, drop the oldest frame (keep latest)
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._frame_queue.put(frame)
                else:
                    # No new frame yet — brief sleep to avoid busy-wait
                    time.sleep(0.001)

        except Exception as e:
            print(f"[ScreenGrabber] Error: {e}")
        finally:
            self._running = False
            if self._camera is not None:
                try:
                    self._camera.stop()
                except Exception:
                    pass
