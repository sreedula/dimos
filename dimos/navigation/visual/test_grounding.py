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

"""Unit tests for the YOLOE grounding fast-path and its VLM fallback.

These are fast, fully stubbed tests: the YOLOE detector and the VL model are
replaced with lightweight fakes, so no model weights, network, or Git-LFS data
are needed. The heavy end-to-end detector behaviour is covered separately by the
``self_hosted`` tests in ``perception/detection/detectors``.
"""

import numpy as np
import pytest

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.navigation.visual.grounding import (
    build_yoloe_grounding_detector,
    ground_candidates_with_yoloe,
    ground_with_position,
    ground_with_yoloe,
    select_by_position,
)
from dimos.navigation.visual.query import get_object_bbox, get_object_bbox_from_image


class _FakeDetection:
    """Stand-in for a Detection2D: only the fields ground_with_yoloe reads."""

    def __init__(self, name: str, confidence: float, bbox: tuple) -> None:
        self.name = name
        self.confidence = confidence
        self.bbox = bbox


class _FakeResult:
    """Stand-in for ImageDetections2D: exposes a ``detections`` list."""

    def __init__(self, detections: list[_FakeDetection]) -> None:
        self.detections = detections


class _FakeDetector:
    """Mimics Yoloe2DDetector.set_prompts/process_image without loading a model."""

    def __init__(self, by_prompt: dict[str, list[_FakeDetection]]) -> None:
        self._by_prompt = by_prompt
        self._prompt: str | None = None
        self.process_calls = 0

    def set_prompts(self, text: list[str]) -> None:
        self._prompt = text[0]

    def process_image(self, image: Image) -> _FakeResult:
        self.process_calls += 1
        return _FakeResult(list(self._by_prompt.get(self._prompt, [])))


class _RaisingVlModel:
    """VL model stub that fails if its (slow) query path is ever taken."""

    def query(self, image, query):  # noqa: ANN001
        raise AssertionError("vl_model.query must not be called on a YOLOE fast-path hit")


class _StubVlModel:
    """VL model stub that records calls and returns a fixed response string."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def query(self, image, query):  # noqa: ANN001
        self.calls += 1
        return self.response


@pytest.fixture
def image() -> Image:
    """A tiny synthetic image; content is irrelevant to the stubbed detector."""
    return Image.from_numpy(np.zeros((8, 8, 3), dtype=np.uint8), format=ImageFormat.BGR)


def test_ground_with_yoloe_returns_bbox_for_present_object(image: Image) -> None:
    detector = _FakeDetector({"person": [_FakeDetection("person", 0.9, (10, 20, 30, 40))]})

    bbox = ground_with_yoloe(detector, image, "person")

    assert bbox == (10.0, 20.0, 30.0, 40.0)
    assert isinstance(bbox, tuple) and len(bbox) == 4
    assert all(isinstance(v, float) for v in bbox)


def test_ground_with_yoloe_returns_highest_confidence_detection(image: Image) -> None:
    detector = _FakeDetector(
        {
            "person": [
                _FakeDetection("person", 0.61, (0, 0, 1, 1)),
                _FakeDetection("person", 0.95, (5, 6, 7, 8)),
                _FakeDetection("person", 0.80, (2, 2, 3, 3)),
            ]
        }
    )

    bbox = ground_with_yoloe(detector, image, "person")

    assert bbox == (5.0, 6.0, 7.0, 8.0)  # the 0.95-confidence box


def test_ground_with_yoloe_returns_none_for_absent_object(image: Image) -> None:
    detector = _FakeDetector({"person": [_FakeDetection("person", 0.9, (1, 2, 3, 4))]})

    # "banana" has no detections -> no-match path.
    assert ground_with_yoloe(detector, image, "banana") is None


def test_get_object_bbox_fast_path_returns_yolo_bbox_without_calling_vlm(image: Image) -> None:
    detector = _FakeDetector({"person": [_FakeDetection("person", 0.9, (10, 20, 30, 40))]})
    vl_model = _RaisingVlModel()  # raises if the VLM is touched

    bbox = get_object_bbox(vl_model, image, "person", detector=detector)

    assert bbox == (10.0, 20.0, 30.0, 40.0)
    assert detector.process_calls >= 1  # YOLOE actually ran


def test_get_object_bbox_falls_back_to_vlm_when_yolo_finds_nothing(image: Image) -> None:
    detector = _FakeDetector({})  # YOLOE finds nothing for any prompt
    vl_model = _StubVlModel('{"name": "banana", "bbox": [1, 2, 3, 4]}')

    bbox = get_object_bbox(vl_model, image, "banana", detector=detector)

    assert vl_model.calls == 1  # fell back to the VLM
    assert bbox == (1.0, 2.0, 3.0, 4.0)


def test_get_object_bbox_without_detector_matches_get_object_bbox_from_image(image: Image) -> None:
    stub = _StubVlModel('{"name": "chair", "bbox": [5, 6, 7, 8]}')

    via_new = get_object_bbox(stub, image, "chair")  # detector=None
    via_original = get_object_bbox_from_image(stub, image, "chair")

    assert via_new == via_original == (5.0, 6.0, 7.0, 8.0)
    assert stub.calls == 2  # both routed through the VLM exactly once each


class _BoomDetector:
    """Detector whose fast path always raises (e.g. model failed to load)."""

    def set_prompts(self, text: list[str]) -> None:
        raise RuntimeError("detector model not loaded")

    def process_image(self, image: Image):  # noqa: ANN201
        raise RuntimeError("detector model not loaded")


def test_get_object_bbox_falls_back_to_vlm_when_detector_raises(image: Image) -> None:
    # A broken fast path must never be worse than the VLM-only baseline.
    vl_model = _StubVlModel('{"name": "person", "bbox": [9, 9, 19, 29]}')

    bbox = get_object_bbox(vl_model, image, "person", detector=_BoomDetector())

    assert vl_model.calls == 1  # exception was swallowed and the VLM was used
    assert bbox == (9.0, 9.0, 19.0, 29.0)


def test_build_yoloe_grounding_detector_returns_none_on_failure(monkeypatch) -> None:
    # If YOLOE can't be constructed (missing weights/encoder), the helper must
    # degrade to None rather than raise, so callers fall back to the VLM.
    import dimos.perception.detection.detectors.yoloe as yoloe_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("weights unavailable")

    monkeypatch.setattr(yoloe_mod, "Yoloe2DDetector", _boom)
    assert build_yoloe_grounding_detector() is None


def test_ground_candidates_returns_all_boxes_sorted_by_confidence(image: Image) -> None:
    detector = _FakeDetector(
        {
            "chair": [
                _FakeDetection("chair", 0.61, (0, 0, 1, 1)),
                _FakeDetection("chair", 0.95, (5, 6, 7, 8)),
                _FakeDetection("chair", 0.80, (2, 2, 3, 3)),
            ]
        }
    )

    candidates = ground_candidates_with_yoloe(detector, image, "chair")

    # Highest confidence first, every detection present, all float 4-tuples.
    assert candidates == [(5.0, 6.0, 7.0, 8.0), (2.0, 2.0, 3.0, 3.0), (0.0, 0.0, 1.0, 1.0)]
    assert all(isinstance(b, tuple) and len(b) == 4 for b in candidates)
    assert all(isinstance(v, float) for b in candidates for v in b)


def test_ground_candidates_returns_empty_list_for_absent_object(image: Image) -> None:
    detector = _FakeDetector({"chair": [_FakeDetection("chair", 0.9, (1, 2, 3, 4))]})

    assert ground_candidates_with_yoloe(detector, image, "banana") == []


def test_select_by_position_returns_none_on_empty() -> None:
    assert select_by_position([], "leftmost") is None


def test_select_by_position_leftmost() -> None:
    boxes = [(50.0, 0.0, 60.0, 10.0), (0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)]
    assert select_by_position(boxes, "leftmost") == (0.0, 0.0, 10.0, 10.0)


def test_select_by_position_rightmost() -> None:
    boxes = [(50.0, 0.0, 60.0, 10.0), (0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)]
    assert select_by_position(boxes, "rightmost") == (50.0, 0.0, 60.0, 10.0)


def test_select_by_position_topmost() -> None:
    boxes = [(0.0, 50.0, 10.0, 60.0), (0.0, 0.0, 10.0, 10.0), (0.0, 20.0, 10.0, 30.0)]
    assert select_by_position(boxes, "topmost") == (0.0, 0.0, 10.0, 10.0)


def test_select_by_position_bottommost() -> None:
    boxes = [(0.0, 50.0, 10.0, 60.0), (0.0, 0.0, 10.0, 10.0), (0.0, 20.0, 10.0, 30.0)]
    assert select_by_position(boxes, "bottommost") == (0.0, 50.0, 10.0, 60.0)


def test_select_by_position_largest() -> None:
    boxes = [(0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 5.0, 5.0)]
    assert select_by_position(boxes, "largest") == (0.0, 0.0, 10.0, 10.0)


def test_select_by_position_smallest() -> None:
    boxes = [(0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 5.0, 5.0)]
    assert select_by_position(boxes, "smallest") == (0.0, 0.0, 2.0, 2.0)


def test_select_by_position_center() -> None:
    # Centers at x=5, x=15, x=100; mean center x ~= 40, so the x=15 box (center
    # at 15) is closest to the centroid of all candidate centers.
    boxes = [(0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0), (95.0, 0.0, 105.0, 10.0)]
    assert select_by_position(boxes, "center") == (10.0, 0.0, 20.0, 10.0)


def test_select_by_position_is_case_insensitive() -> None:
    boxes = [(50.0, 0.0, 60.0, 10.0), (0.0, 0.0, 10.0, 10.0)]
    assert select_by_position(boxes, "LeftMost") == (0.0, 0.0, 10.0, 10.0)


def test_ground_with_position_with_qualifier_resolves_geometrically(image: Image) -> None:
    detector = _FakeDetector(
        {
            "chair": [
                _FakeDetection("chair", 0.95, (50, 0, 60, 10)),  # top confidence, rightmost
                _FakeDetection("chair", 0.80, (0, 0, 10, 10)),  # leftmost
            ]
        }
    )

    # The qualifier overrides confidence: leftmost wins despite lower confidence.
    assert ground_with_position(detector, image, "chair", qualifier="leftmost") == (
        0.0,
        0.0,
        10.0,
        10.0,
    )


def test_ground_with_position_without_qualifier_returns_top_confidence(image: Image) -> None:
    detector = _FakeDetector(
        {
            "chair": [
                _FakeDetection("chair", 0.80, (0, 0, 10, 10)),
                _FakeDetection("chair", 0.95, (50, 0, 60, 10)),
            ]
        }
    )

    assert ground_with_position(detector, image, "chair") == (50.0, 0.0, 60.0, 10.0)


def test_ground_with_position_returns_none_for_absent_object(image: Image) -> None:
    detector = _FakeDetector({"chair": [_FakeDetection("chair", 0.9, (1, 2, 3, 4))]})

    assert ground_with_position(detector, image, "banana", qualifier="leftmost") is None
    assert ground_with_position(detector, image, "banana") is None
