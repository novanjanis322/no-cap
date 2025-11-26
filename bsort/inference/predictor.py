"""Inference pipeline for bottle cap detection."""

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class Predictor:
    """Predictor for bottle cap detection inference."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize predictor with configuration.

        Args:
            config: Inference configuration dictionary.
        """
        self.config = config
        # Load model
        model_path = config["model"]["checkpoint_path"]
        self.model = YOLO(model_path)
        # Set device
        self.device = config["inference"].get("device", "cuda")
        self.model.to(self.device)
        # Inference settings
        self.conf_threshold = config["inference"].get("confidence_threshold", 0.25)
        self.iou_threshold = config["inference"].get("iou_threshold", 0.45)
        self.input_size = config["model"].get("input_size", 640)
        # Class names
        self.class_names = config.get(
            "classes", {0: "light_blue", 1: "dark_blue", 2: "others"}
        )
        # Warmup
        if config["performance"].get("measure_latency", False):
            self._warmup()

    def _warmup(self) -> None:
        """Warmup the model for accurate latency measurement."""
        warmup_iterations = self.config["performance"].get("warmup_iterations", 10)
        print(f"Warming up model for {warmup_iterations} iterations...")
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(warmup_iterations):
            self.model.predict(
                dummy_input,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False,
            )
        print("Warmup completed")

    def predict(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference on an image.

        Args:
            image_path: Path to input image.
            output_path: Optional path to save output image with predictions.

        Returns:
            Dictionary containing detections and inference time.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Run inference with timing
        start_time = time.perf_counter()
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False,
        )
        end_time = time.perf_counter()
        inference_time = (
            end_time - start_time
        ) * 1000  # Convert to ms        # Parse results
        detections = self._parse_results(results[0])
        # Save output if requested
        if output_path:
            from bsort.inference.visualizer import Visualizer

            visualizer = Visualizer(self.config)
            visualizer.draw_predictions(image, detections, output_path)
        return {
            "detections": detections,
            "inference_time": inference_time,
        }

    def benchmark(self, image_path: str) -> Dict[str, float]:
        """Benchmark inference speed on an image.

        Args:
            image_path: Path to input image.

        Returns:
            Dictionary with timing statistics.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        iterations = self.config["performance"].get("benchmark_iterations", 100)
        times = []
        print(f"Benchmarking inference speed for {iterations} iterations...")
        for _ in range(iterations):
            start_time = time.perf_counter()
            self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False,
            )
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
        }

    def _parse_results(self, result) -> List[Dict[str, Any]]:
        """Parse YOLO results into structured format.

        Args:
            result: YOLO result object.

        Returns:
            List of detection dictionaries.
        """
        detections = []
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": self.class_names.get(cls, "unknown"),
                }
            )
        return detections
