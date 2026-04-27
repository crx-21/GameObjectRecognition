# overlay_window.py — Module 3: Overlay Window (Consumer)
# PyQt5 frameless, transparent, always-on-top, click-through window.
# Draws color-coded bounding boxes and confidence labels over the game.

import queue
import ctypes
import time

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QFont

import win32con
import win32gui
import win32process

import config


class OverlayWindow(QWidget):
    """
    Consumer module — Transparent, click-through overlay that draws
    bounding boxes and confidence labels on top of the game window.
    """

    # CS2 window class name (Source 2 engine)
    CS2_WINDOW_CLASS = "Valve001"
    CS2_WINDOW_TITLE = "Counter-Strike 2"

    def __init__(self, results_queue: queue.Queue):
        """
        Args:
            results_queue: Queue to read detection results from.
        """
        super().__init__()
        self._results_queue = results_queue
        self._detections: list = []
        self._game_hwnd = None

        self._init_window()
        self._init_timer()
        self._find_game_window()

    def _init_window(self):
        """Configure the overlay as frameless, transparent, always-on-top, click-through."""
        # Frameless + transparent + always-on-top
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # Hides from taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # Start with full-screen geometry, will be adjusted to match game window
        screen = QApplication.primaryScreen()
        geometry = screen.geometry()
        self.setGeometry(geometry)

        self.setWindowTitle("CS2 Enemy Recon Overlay")

    def _find_game_window(self):
        """Find the CS2 game window handle and store its HWND."""
        try:
            # Try to find by window class first (most reliable)
            def callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    try:
                        class_name = win32gui.GetClassName(hwnd)
                        window_title = win32gui.GetWindowText(hwnd)
                        # Check for Source 2 engine window
                        if class_name == self.CS2_WINDOW_CLASS:
                            # Verify it's CS2 by checking the process name
                            _, pid = win32process.GetWindowThreadProcessId(hwnd)
                            import psutil
                            try:
                                proc = psutil.Process(pid)
                                if "cs2" in proc.name().lower():
                                    self._game_hwnd = hwnd
                                    return False  # Stop enumeration
                            except (psutil.NoSuchProcess, Exception):
                                pass
                    except Exception:
                        pass
                return True  # Continue enumeration

            win32gui.EnumWindows(callback, None)

            # Fallback: try to find by window title if class search failed
            if self._game_hwnd is None:
                self._game_hwnd = win32gui.FindWindow(None, self.CS2_WINDOW_TITLE)

            if self._game_hwnd:
                print(f"[Overlay] Found CS2 window (HWND: {self._game_hwnd})")
                self._update_geometry_to_match_game()
            else:
                print("[Overlay] CS2 window not found - using full screen")

        except Exception as e:
            print(f"[Overlay] Error finding game window: {e}")

    def _update_geometry_to_match_game(self):
        """Update overlay geometry to match the game window."""
        if not self._game_hwnd:
            return

        try:
            left, top, right, bottom = win32gui.GetWindowRect(self._game_hwnd)
            width = right - left
            height = bottom - top
            self.setGeometry(left, top, width, height)
        except Exception as e:
            print(f"[Overlay] Error updating geometry: {e}")

    def _refresh_game_window_position(self):
        """Periodically refresh the overlay position to match the game window."""
        if not self._game_hwnd:
            # Try to find the game window again if it wasn't found before
            self._find_game_window()
            return

        try:
            # Check if the game window still exists
            if not win32gui.IsWindow(self._game_hwnd):
                print("[Overlay] Game window closed, searching for new window...")
                self._game_hwnd = None
                self._find_game_window()
                return

            # Update position to match game window
            self._update_geometry_to_match_game()
        except Exception as e:
            print(f"[Overlay] Error refreshing position: {e}")

    def showEvent(self, event):
        """After the window is shown, apply click-through via Win32 API."""
        super().showEvent(event)
        self._set_click_through()

    def _set_click_through(self):
        """Make the window click-through using Windows extended styles."""
        hwnd = int(self.winId())
        # Get current extended style
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        # Add WS_EX_TRANSPARENT + WS_EX_LAYERED to make click-through
        win32gui.SetWindowLong(
            hwnd,
            win32con.GWL_EXSTYLE,
            ex_style | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED,
        )

    def _init_timer(self):
        """Set up timers to poll the results queue and refresh window position."""
        # Timer for updating detections (fast)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_detections)
        self._timer.start(config.OVERLAY_REFRESH_MS)

        # Timer for refreshing game window position (slower, every 500ms)
        self._position_timer = QTimer(self)
        self._position_timer.timeout.connect(self._refresh_game_window_position)
        self._position_timer.start(500)

    def _update_detections(self):
        """Pull the latest detections from the queue and schedule a repaint."""
        try:
            # Drain to the most recent result
            latest = None
            while not self._results_queue.empty():
                latest = self._results_queue.get_nowait()
            if latest is not None:
                self._detections = latest
                self.update()  # Triggers paintEvent
        except Exception:
            pass

    def paintEvent(self, event):
        """Draw bounding boxes and labels for each detection."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        font = QFont("Consolas", config.FONT_SIZE, QFont.Bold)
        painter.setFont(font)

        for det in self._detections:
            x1, y1, x2, y2 = det.bbox
            r, g, b = config.get_color(det.class_id)
            color = QColor(r, g, b)
            confidence_pct = int(det.confidence * 100)

            # --- Draw bounding box ---
            pen = QPen(color, config.BOX_THICKNESS)
            painter.setPen(pen)
            painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

            # --- Draw label background ---
            label = f"{det.class_name} {confidence_pct}%"
            font_metrics = painter.fontMetrics()
            text_width = font_metrics.horizontalAdvance(label) + 8
            text_height = font_metrics.height() + 4

            bg_color = QColor(r, g, b, 160)  # Semi-transparent background
            painter.fillRect(
                QRect(x1, y1 - text_height, text_width, text_height),
                bg_color,
            )

            # --- Draw label text ---
            painter.setPen(QColor(255, 255, 255))  # White text
            painter.drawText(x1 + 4, y1 - 4, label)

        painter.end()
