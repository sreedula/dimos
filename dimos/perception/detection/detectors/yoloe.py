# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLOE  # type: ignore[attr-defined]

from dimos.msgs.sensor_msgs.Image import Image
from dimos.perception.detection.detectors.base import Detector
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.utils.data import get_data
from dimos.utils.gpu_utils import is_cuda_available
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class YoloePromptMode(Enum):
    """YOLO-E prompt modes."""

    LRPC = "lrpc"
    PROMPT = "prompt"


class Yoloe2DDetector(Detector):
    def __init__(
        self,
        model_path: str = "models_yoloe",
        model_name: str | None = None,
        device: str | None = None,
        prompt_mode: YoloePromptMode = YoloePromptMode.LRPC,
        exclude_class_ids: list[int] | None = None,
        max_area_ratio: float | None = 0.3,
        confidence: float = 0.6,
        iou_threshold: float = 0.6,
    ) -> None:
        """
        Initialize YOLO-E 2D detector.

        Args:
            model_path: Path to model directory (fetched via get_data from LFS).
            model_name: Model filename. Defaults based on prompt_mode.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
            prompt_mode: LRPC for prompt-free detection, PROMPT for text/visual prompting.
            exclude_class_ids: Class IDs to filter out from results (pass [] to disable).
            max_area_ratio: Maximum bbox area ratio (0-1) relative to image.
            confidence: Minimum detection confidence (0-1]. Lower values raise
                recall (find smaller/occluded objects) at the cost of precision.
            iou_threshold: NMS IoU threshold (0-1]. Higher values keep more
                overlapping boxes (useful for crowded scenes / touching objects).
        """
        if model_name is None:
            if prompt_mode == YoloePromptMode.LRPC:
                model_name = "yoloe-11s-seg-pf.pt"
            else:
                model_name = "yoloe-11s-seg.pt"

        self.model = YOLOE(get_data(model_path) / model_name)
        self.prompt_mode = prompt_mode
        self._visual_prompts: dict[str, NDArray[Any]] | None = None
        self.max_area_ratio = max_area_ratio
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self._lock = threading.Lock()

        if prompt_mode == YoloePromptMode.PROMPT:
            self.set_prompts(text=["nothing"])
        self.exclude_class_ids = set(exclude_class_ids) if exclude_class_ids else set()

        if self.max_area_ratio is not None and not (0.0 < self.max_area_ratio <= 1.0):
            raise ValueError("max_area_ratio must be in the range (0, 1].")

        if not (0.0 < self.confidence <= 1.0):
            raise ValueError("confidence must be in the range (0, 1].")

        if not (0.0 < self.iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in the range (0, 1].")

        if device:
            self.device = device
        elif is_cuda_available():  # type: ignore[no-untyped-call]
            self.device = "cuda"
        else:
            # NOTE: Apple-Silicon MPS is deliberately NOT auto-selected. YOLOE's
            # MobileCLIP text-prompt encoder loads via torch.jit with float64
            # weights, which MPS cannot convert ("MPS doesn't support float64"),
            # so set_prompts would fail after the first inference moves the model
            # to MPS. CPU is correct and reliable for the text-prompt path.
            self.device = "cpu"

    def set_prompts(
        self,
        text: list[str] | None = None,
        bboxes: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Set prompts for detection. Provide either text or bboxes, not both.

        Args:
            text: List of class names to detect.
            bboxes: Bounding boxes in xyxy format, shape (N, 4).
        """
        if text is not None and bboxes is not None:
            raise ValueError("Provide either text or bboxes, not both.")
        if text is None and bboxes is None:
            raise ValueError("Must provide either text or bboxes.")

        with self._lock:
            self.model.predictor = None
            if text is not None:
                self.model.set_classes(text, self.model.get_text_pe(text))  # type: ignore[no-untyped-call]
                self._visual_prompts = None
            else:
                cls = np.arange(len(bboxes), dtype=np.int16)  # type: ignore[arg-type]
                self._visual_prompts = {"bboxes": bboxes, "cls": cls}  # type: ignore[dict-item]

    def process_image(self, image: Image) -> "ImageDetections2D[Any]":
        """
        Process an image and return detection results.

        Args:
            image: Input image

        Returns:
            ImageDetections2D containing all detected objects
        """
        source = image.to_opencv()

        with self._lock:
            if self._visual_prompts is not None:
                # Visual (bbox) prompts need YOLOE's dedicated VP predictor. The
                # standard track() path mishandles their output (NMS receives a
                # tuple, not a tensor) and crashes; predict() with the seg VP
                # predictor — matching the -seg checkpoint these models load — is
                # the supported path.
                from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

                # The VP predictor rebinds the inner model's `names` to a list,
                # which breaks a later text set_prompts() (it expects a dict).
                # Save and restore it so the detector can switch text <-> bbox.
                inner_model = self.model.model
                saved_names = inner_model.names
                try:
                    results = self.model.predict(
                        source=source,
                        device=self.device,
                        conf=self.confidence,
                        iou=self.iou_threshold,
                        verbose=False,
                        visual_prompts=self._visual_prompts,
                        predictor=YOLOEVPSegPredictor,
                    )
                finally:
                    inner_model.names = saved_names
            else:
                track_kwargs = {
                    "source": source,
                    "device": self.device,
                    "conf": self.confidence,
                    "iou": self.iou_threshold,
                    "persist": True,
                    "verbose": False,
                }
                try:
                    results = self.model.track(**track_kwargs)  # type: ignore[arg-type]
                except Exception:
                    # A GPU/MPS backend op can be unsupported for a given input;
                    # fall back to CPU once (permanently for this detector)
                    # instead of failing. CPU is the baseline that always works.
                    if self.device == "cpu":
                        raise
                    logger.warning(
                        "YOLOE inference on '%s' failed; falling back to CPU.",
                        self.device,
                        exc_info=True,
                    )
                    self.device = "cpu"
                    track_kwargs["device"] = "cpu"
                    results = self.model.track(**track_kwargs)  # type: ignore[arg-type]

        detections = ImageDetections2D.from_ultralytics_result(image, results)
        return self._apply_filters(image, detections)

    def _apply_filters(
        self,
        image: Image,
        detections: "ImageDetections2D[Any]",
    ) -> "ImageDetections2D[Any]":
        if not self.exclude_class_ids and self.max_area_ratio is None:
            return detections

        predicates = []

        if self.exclude_class_ids:
            predicates.append(lambda det: det.class_id not in self.exclude_class_ids)

        if self.max_area_ratio is not None:
            image_area = image.width * image.height

            def area_filter(det):  # type: ignore[no-untyped-def]
                if image_area <= 0:
                    return True
                return (det.bbox_2d_volume() / image_area) <= self.max_area_ratio

            predicates.append(area_filter)

        filtered = detections.detections
        for predicate in predicates:
            filtered = [det for det in filtered if predicate(det)]  # type: ignore[no-untyped-call]

        return ImageDetections2D(image, filtered)

    def stop(self) -> None:
        """Clean up resources used by the detector."""
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            predictor = self.model.predictor
            if hasattr(predictor, "trackers") and predictor.trackers:
                for tracker in predictor.trackers:
                    if hasattr(tracker, "tracker") and hasattr(tracker.tracker, "gmc"):
                        gmc = tracker.tracker.gmc
                        if hasattr(gmc, "executor") and gmc.executor is not None:
                            gmc.executor.shutdown(wait=True)
            self.model.predictor = None
