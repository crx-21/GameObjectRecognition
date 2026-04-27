# fps_counter.py — FPS Tracking Utility
# Simple frame-rate counter using time.perf_counter().

import time


class FPSCounter:
    """Tracks frames-per-second using a sliding window."""

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames to average over.
        """
        self._window_size = window_size
        self._timestamps: list[float] = []
        self._fps: float = 0.0

    def tick(self) -> float:
        """Call once per frame. Returns the current FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)

        # Keep only the last `window_size` timestamps
        while len(self._timestamps) > self._window_size:
            self._timestamps.pop(0)

        # Need at least 2 timestamps to compute FPS
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

        return self._fps

    @property
    def fps(self) -> float:
        """Return the last computed FPS value."""
        return self._fps

    def reset(self):
        """Clear all recorded timestamps."""
        self._timestamps.clear()
        self._fps = 0.0
