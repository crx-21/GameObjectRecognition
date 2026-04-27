# inference_engine.py — Module 2: Inference Engine (Worker)
# Pulls frames from the capture queue, runs YOLOv8 inference,
# filters by confidence, and pushes detection results to a results queue.

import threading
import queue
import time

from ultralytics import YOLO
import numpy as np

import config
from utils.fps_counter import FPSCounter


class Detection:
    """Represents a single detected entity."""

    __slots__ = ("bbox", "confidence", "class_id", "class_name")

    def __init__(self, bbox: tuple, confidence: float, class_id: int, class_name: str):
        """
        Args:
            bbox: (x1, y1, x2, y2) pixel coordinates of the bounding box.
            confidence: Detection confidence score (0.0 – 1.0).
            class_id: Integer class ID from the model.
            class_name: Human-readable class label.
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "bbox": self.bbox,
            "confidence": int(self.confidence * 10000) / 10000,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class InferenceEngine:
    """
    Worker module — pulls frames from the capture queue, runs YOLOv8
    inference, and pushes structured detection results to the results queue.
    """

    def __init__(self, frame_queue: queue.Queue, results_queue: queue.Queue):
        """
        Args:
            frame_queue: Queue to pull captured frames from (producer).
            results_queue: Queue to push detection results into (consumer).
        """
        self._frame_queue = frame_queue
        self._results_queue = results_queue
        self._running = False
        self._thread: threading.Thread | None = None
        self._model: YOLO | None = None
        self._frame_counter = 0  # For frame skipping
        self._fps_counter = FPSCounter()
        self._last_fps_print = time.time()

    def start(self):
        """Load the YOLO model and start the inference thread."""
        print(f"[InferenceEngine] Loading model: {config.MODEL_PATH}")
        print(f"[InferenceEngine] Device: {config.DEVICE}")
        self._model = YOLO(config.MODEL_PATH)
        self._running = True
        thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread = thread
        thread.start()

    def stop(self):
        """Signal the inference thread to stop and wait for it to finish."""
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)

    @property
    def is_running(self) -> bool:
        return self._running

    def _inference_loop(self):
        """Main inference loop — runs on a daemon thread."""
        try:
            while self._running:
                # Pull a frame from the capture queue (blocking with timeout)
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Frame skipping for performance - only run inference on every Nth frame
                self._frame_counter += 1
                if self._frame_counter % config.INFERENCE_SKIP_RATE != 0:
                    # Skip this frame, reuse last known detections
                    continue

                # Run YOLOv8 inference with optimized settings
                results = self._model.predict(
                    source=frame,
                    conf=config.CONFIDENCE_THRESHOLD,
                    iou=config.IOU_THRESHOLD,
                    imgsz=config.MODEL_INPUT_SIZE,
                    device=config.DEVICE,
                    verbose=False,
                    half=True if config.DEVICE == "cuda" else False,  # FP16 inference on GPU
                )

                # Parse the results into Detection objects
                detections = self._parse_results(results)

                # Update FPS counter
                self._fps_counter.tick()

                # Print FPS every 5 seconds
                now = time.time()
                if now - self._last_fps_print >= 5.0:
                    fps = self._fps_counter.fps
                    print(f"[InferenceEngine] FPS: {fps:.1f} | Detections: {len(detections)}")
                    self._last_fps_print = now

                # Push detections to the results queue (drop old if full)
                if self._results_queue.full():
                    try:
                        self._results_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._results_queue.put(detections)

        except Exception as e:
            print(f"[InferenceEngine] Error: {e}")
        finally:
            self._running = False

    def _parse_results(self, results) -> list[Detection]:
        """
        Parse YOLO results into a list of Detection objects.

        Args:
            results: Ultralytics YOLO results object.

        Returns:
            List of Detection instances.
        """
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Bounding box coordinates (top-left x, top-left y, bottom-right x, bottom-right y)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = config.get_class_name(class_id)

                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        return detections
