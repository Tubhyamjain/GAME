from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


VEHICLE_LABELS = {
    "car",
    "motorbike",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "van",
    "train",
}


@dataclass
class CalibrationConfig:
    meters_per_pixel: float = 0.05
    image_points: Optional[np.ndarray] = None  # shape (4,2)
    world_points: Optional[np.ndarray] = None  # shape (4,2) in meters


class PerspectiveMapper:
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.homography = None
        if (
            config.image_points is not None
            and config.world_points is not None
            and len(config.image_points) == 4
            and len(config.world_points) == 4
        ):
            self.homography, _ = cv2.findHomography(
                np.asarray(config.image_points, dtype=np.float32),
                np.asarray(config.world_points, dtype=np.float32),
            )

    def to_world(self, p: Tuple[float, float]) -> np.ndarray:
        if self.homography is None:
            return np.asarray(p, dtype=np.float32) * self.config.meters_per_pixel

        px = np.array([[[p[0], p[1]]]], dtype=np.float32)
        world = cv2.perspectiveTransform(px, self.homography)[0][0]
        return world


class SpeedEstimator:
    def __init__(self, mapper: PerspectiveMapper, smoothing_window: int = 5):
        self.mapper = mapper
        self.history: Dict[int, List[Tuple[float, np.ndarray]]] = {}
        self.window = smoothing_window

    def update(self, track_id: int, center: Tuple[float, float], t_sec: float) -> Optional[float]:
        world = self.mapper.to_world(center)
        bucket = self.history.setdefault(track_id, [])
        bucket.append((t_sec, world))
        if len(bucket) > self.window + 1:
            bucket.pop(0)

        if len(bucket) < 2:
            return None

        speeds_kmh: List[float] = []
        for (t1, p1), (t2, p2) in zip(bucket[:-1], bucket[1:]):
            dt = t2 - t1
            if dt <= 1e-6:
                continue
            dist_m = np.linalg.norm(p2 - p1)
            speed_mps = dist_m / dt
            speeds_kmh.append(speed_mps * 3.6)

        if not speeds_kmh:
            return None

        arr = np.asarray(speeds_kmh, dtype=np.float32)
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        iqr = max(q3 - q1, 1e-6)
        clipped = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
        if clipped.size == 0:
            clipped = arr
        return float(np.median(clipped))


class PlateRecognizer:
    def __init__(self, min_confidence: float = 0.35):
        self._ocr_reader = None
        self.min_confidence = min_confidence

    def _ensure_reader(self):
        if self._ocr_reader is None:
            import easyocr

            self._ocr_reader = easyocr.Reader(["en"], gpu=False)

    def detect_plate_text(self, crop: np.ndarray) -> Optional[str]:
        if crop.size == 0:
            return None

        # Basic heuristic: run OCR directly on grayscale high-contrast crop.
        self._ensure_reader()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        results = self._ocr_reader.readtext(gray)
        if not results:
            return None

        best = max(results, key=lambda x: x[2])
        if float(best[2]) < self.min_confidence:
            return None

        text = best[1].strip()
        text = re.sub(r"[^A-Z0-9-]", "", text.upper())
        if len(text) < 4:
            return None
        return text


class VehicleAnalyzer:
    def __init__(
        self,
        confidence: float = 0.25,
        calibration: Optional[CalibrationConfig] = None,
        enable_plate_ocr: bool = False,
    ):
        self.model = YOLO("yolov8n.pt")
        self.confidence = confidence
        self.mapper = PerspectiveMapper(calibration or CalibrationConfig())
        self.speed_estimator = SpeedEstimator(self.mapper)
        self.enable_plate_ocr = enable_plate_ocr
        self.plate_recognizer = PlateRecognizer() if enable_plate_ocr else None

    @staticmethod
    def _label_of(class_id: int, names: Dict[int, str]) -> str:
        return names.get(class_id, str(class_id)).lower()

    @staticmethod
    def _center(xyxy: Iterable[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def process_video(self, input_path: Path, output_path: Path) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        logs: List[Dict[str, object]] = []

        for frame_idx, result in enumerate(
            self.model.track(
                source=str(input_path),
                stream=True,
                conf=self.confidence,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )
        ):
            source_frame = result.orig_img
            frame = source_frame.copy()
            timestamp = frame_idx / fps
            boxes = result.boxes
            names = result.names

            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else np.arange(len(xyxy))
                )

                for bb, cid, tid in zip(xyxy, cls, ids):
                    label = self._label_of(cid, names)
                    if label not in VEHICLE_LABELS:
                        continue

                    x1, y1, x2, y2 = map(int, bb)
                    x1 = max(0, min(x1, width - 1))
                    x2 = max(0, min(x2, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    y2 = max(0, min(y2, height - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    center = self._center(bb)
                    speed = self.speed_estimator.update(int(tid), center, timestamp)
                    speed_text = f"{speed:.1f} km/h" if speed is not None else "estimating..."

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID {tid} | {label} | {speed_text}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    plate_text = None
                    if self.enable_plate_ocr and self.plate_recognizer is not None:
                        crop = source_frame[y1:y2, x1:x2]
                        plate_text = self.plate_recognizer.detect_plate_text(crop)
                        if plate_text:
                            cv2.putText(
                                frame,
                                f"Plate: {plate_text}",
                                (x1, min(height - 10, y2 + 18)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                2,
                            )

                    logs.append(
                        {
                            "timestamp_sec": round(timestamp, 2),
                            "track_id": int(tid),
                            "vehicle_type": label,
                            "speed_kmh": None if speed is None else round(speed, 2),
                            "speed_range": self._speed_range(speed),
                            "plate_text": plate_text,
                        }
                    )

            writer.write(frame)

        cap.release()
        writer.release()
        return pd.DataFrame(logs)

    @staticmethod
    def _speed_range(speed_kmh: Optional[float]) -> str:
        if speed_kmh is None:
            return "unknown"
        if speed_kmh < 20:
            return "slow"
        if speed_kmh < 50:
            return "moderate"
        if speed_kmh < 90:
            return "fast"
        return "very fast"
