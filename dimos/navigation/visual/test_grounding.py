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
from dimos.navigation.visual import grounding as grounding_mod
from dimos.navigation.visual.grounding import (
    box_iou,
    build_yoloe_grounding_detector,
    ground_candidates_with_yoloe,
    ground_with_attribute,
    ground_with_position,
    ground_with_tracking,
    ground_with_yoloe,
    parse_grounding_query,
    parse_relational_query,
    resolve_grounding,
    select_by_clip,
    select_by_nearest_to_reference,
    select_by_position,
    select_by_relation,
    select_nearest,
    singularize,
)
from dimos.navigation.visual.query import (
    get_object_bbox,
    get_object_bbox_from_image,
    get_object_bboxes,
)


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

    def query(self, image, query):
        raise AssertionError("vl_model.query must not be called on a YOLOE fast-path hit")


class _StubVlModel:
    """VL model stub that records calls and returns a fixed response string."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def query(self, image, query):
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


class _ModeFakeDetector:
    """Fake that tracks text vs visual-prompt mode, like the real detector."""

    def __init__(self) -> None:
        self._visual_prompts = None
        self._prompt: str | None = None

    def set_prompts(self, text: list[str] | None = None, bboxes=None) -> None:
        if bboxes is not None:
            self._visual_prompts = {"bboxes": bboxes}
            self._prompt = None
        else:
            self._visual_prompts = None
            self._prompt = text[0]

    def process_image(self, image: Image) -> _FakeResult:
        if self._visual_prompts is not None:
            return _FakeResult([_FakeDetection("visual", 0.9, (0, 0, 1, 1))])
        return _FakeResult([_FakeDetection(self._prompt or "", 0.9, (5, 6, 7, 8))])


def test_ground_candidates_resets_text_when_detector_switched_to_visual(image: Image) -> None:
    detector = _ModeFakeDetector()
    # Ground a text prompt, populating the memo.
    assert ground_candidates_with_yoloe(detector, image, "person") == [(5.0, 6.0, 7.0, 8.0)]
    # Something else switches the detector to a visual (bbox) prompt.
    detector.set_prompts(bboxes=[[1, 2, 3, 4]])
    # Re-grounding the SAME text must re-set the text prompt, not return the
    # stale visual-prompt detection.
    assert ground_candidates_with_yoloe(detector, image, "person") == [(5.0, 6.0, 7.0, 8.0)]


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

    def process_image(self, image: Image):
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


# --- CLIP attribute re-ranking (Phase 2) ----------------------------------
#
# These stay fully stubbed: an injected fake scorer stands in for CLIP, so no
# weights, network, or torch are touched. The real CLIP path is exercised by the
# self_hosted test at the bottom of this file.


def test_select_by_clip_picks_argmax_candidate_via_injected_scorer(image: Image) -> None:
    boxes = [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0), (4.0, 4.0, 5.0, 5.0)]

    # Fake scorer: the middle box scores highest, so it must be chosen — without
    # loading CLIP at all.
    def fake_scorer(img, candidates, phrase):
        assert candidates == boxes
        return [0.1, 0.9, 0.3]

    assert select_by_clip(image, boxes, "red mug", scorer=fake_scorer) == (2.0, 2.0, 3.0, 3.0)


def test_select_by_clip_returns_none_on_empty(image: Image) -> None:
    # No candidates -> nothing to rank. The (default) scorer must never be called.
    def _boom(img, candidates, phrase):
        raise AssertionError("scorer must not run on empty candidates")

    assert select_by_clip(image, [], "red mug", scorer=_boom) is None


def test_ground_with_attribute_without_phrase_returns_top_confidence(image: Image) -> None:
    detector = _FakeDetector(
        {
            "mug": [
                _FakeDetection("mug", 0.80, (0, 0, 10, 10)),
                _FakeDetection("mug", 0.95, (50, 0, 60, 10)),
            ]
        }
    )

    # No phrase -> behaves like ground_with_yoloe: top-confidence box wins.
    assert ground_with_attribute(detector, image, "mug") == (50.0, 0.0, 60.0, 10.0)


def test_ground_with_attribute_with_phrase_reranks_via_clip(image: Image, monkeypatch) -> None:
    detector = _FakeDetector(
        {
            "mug": [
                _FakeDetection("mug", 0.95, (50, 0, 60, 10)),  # top confidence
                _FakeDetection("mug", 0.80, (0, 0, 10, 10)),  # lower confidence
            ]
        }
    )

    # Stub the module-level clip_scores so select_by_clip re-ranks without CLIP:
    # the lower-confidence box scores highest and so must override confidence.
    def fake_clip_scores(img, candidates, phrase):
        assert phrase == "red mug"
        return [0.2, 0.8]

    monkeypatch.setattr("dimos.navigation.visual.grounding.clip_scores", fake_clip_scores)

    assert ground_with_attribute(detector, image, "mug", phrase="red mug") == (
        0.0,
        0.0,
        10.0,
        10.0,
    )


def test_ground_with_attribute_returns_none_for_absent_object(image: Image) -> None:
    detector = _FakeDetector({"mug": [_FakeDetection("mug", 0.9, (1, 2, 3, 4))]})

    assert ground_with_attribute(detector, image, "banana") is None
    assert ground_with_attribute(detector, image, "banana", phrase="red banana") is None


# --- Track-consistency disambiguation (Phase 3) ---------------------------
#
# These prove the grounded box stays locked onto the same instance across
# frames by preferring the candidate nearest the target's last known box,
# instead of letting confidence flip it to a different instance.


def test_box_iou_identical_boxes_is_one() -> None:
    box = (10.0, 20.0, 30.0, 50.0)
    assert box_iou(box, box) == 1.0


def test_box_iou_disjoint_boxes_is_zero() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 110.0, 110.0)
    assert box_iou(a, b) == 0.0


def test_box_iou_partial_overlap_known_value() -> None:
    # Two 10x10 boxes offset by (5, 5): intersection is the 5x5 corner = 25,
    # union is 100 + 100 - 25 = 175, so IoU = 25/175 = 1/7.
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    assert box_iou(a, b) == pytest.approx(25.0 / 175.0)


def test_select_nearest_returns_none_on_empty() -> None:
    assert select_nearest([], (0.0, 0.0, 10.0, 10.0)) is None


def test_select_nearest_prefers_overlapping_target_over_distant_box() -> None:
    # The target was last here. The "same target slightly moved" candidate
    # overlaps it strongly; the "different instance" is far away with zero
    # overlap. Listed different-instance-first (as if it were higher
    # confidence) to prove IoU — not order — drives the choice.
    prev = (100.0, 100.0, 140.0, 140.0)
    different_instance = (300.0, 300.0, 340.0, 340.0)
    moved_same_target = (104.0, 101.0, 144.0, 141.0)

    assert select_nearest([different_instance, moved_same_target], prev) == moved_same_target


def test_select_nearest_falls_back_to_nearest_center_when_no_overlap() -> None:
    # No candidate overlaps the last box (all IoU 0): track through the gap by
    # picking the nearest center. The near box wins over the far one.
    prev = (0.0, 0.0, 10.0, 10.0)
    near = (20.0, 20.0, 30.0, 30.0)  # center (25, 25)
    far = (200.0, 200.0, 210.0, 210.0)  # center (205, 205)

    assert select_nearest([far, near], prev) == near


def test_select_nearest_returns_none_when_best_iou_below_min_iou() -> None:
    # Only a sliver of overlap; a strict min_iou treats the target as lost.
    prev = (0.0, 0.0, 10.0, 10.0)
    sliver = (8.0, 8.0, 18.0, 18.0)  # IoU ~= 0.02

    assert select_nearest([sliver], prev, min_iou=0.5) is None


def test_ground_with_tracking_locks_onto_nearest_not_top_confidence(image: Image) -> None:
    # Two people: the top-confidence detection is a *different* instance far from
    # where we last saw our target; the lower-confidence detection is our target
    # slightly moved. With prev_box, tracking must keep our target.
    detector = _FakeDetector(
        {
            "person": [
                _FakeDetection("person", 0.95, (300, 300, 340, 340)),  # top conf, different
                _FakeDetection("person", 0.80, (104, 101, 144, 141)),  # our moved target
            ]
        }
    )
    prev = (100.0, 100.0, 140.0, 140.0)

    assert ground_with_tracking(detector, image, "person", prev) == (104.0, 101.0, 144.0, 141.0)


def test_ground_with_tracking_without_prev_box_returns_top_confidence(image: Image) -> None:
    detector = _FakeDetector(
        {
            "person": [
                _FakeDetection("person", 0.95, (300, 300, 340, 340)),
                _FakeDetection("person", 0.80, (104, 101, 144, 141)),
            ]
        }
    )

    # No prev_box -> behaves like ground_with_yoloe: top-confidence box wins.
    assert ground_with_tracking(detector, image, "person") == (300.0, 300.0, 340.0, 340.0)


def test_ground_with_tracking_returns_none_for_absent_object(image: Image) -> None:
    detector = _FakeDetector({"person": [_FakeDetection("person", 0.9, (1, 2, 3, 4))]})

    assert ground_with_tracking(detector, image, "banana") is None
    assert ground_with_tracking(detector, image, "banana", (0.0, 0.0, 10.0, 10.0)) is None


@pytest.mark.self_hosted
def test_clip_reranks_real_bus_crops() -> None:
    """Real CLIP end-to-end: re-rank two literal crops of the ultralytics bus.jpg.

    Deselected from the default suite (self_hosted): it loads the actual CLIP
    weights and proves the real re-ranking path picks the crop matching the
    phrase. The image is split into a left and a right half, and CLIP is asked
    which better matches "a red bus" — the bus dominates the left/center of the
    asset, so that crop must win.
    """
    import cv2
    from ultralytics.utils import ASSETS

    from dimos.navigation.visual.grounding import clip_scores, select_by_clip

    bgr = cv2.imread(str(ASSETS / "bus.jpg"))
    assert bgr is not None, "bus.jpg asset not found"
    h, w = bgr.shape[:2]
    img = Image.from_numpy(bgr, format=ImageFormat.BGR)

    left = (0.0, 0.0, w / 2.0, float(h))  # the red bus
    right = (w / 2.0, 0.0, float(w), float(h))  # mostly the person on the right
    candidates = [left, right]

    scores = clip_scores(img, candidates, "a red bus")
    assert len(scores) == 2
    assert scores[0] > scores[1]  # the bus half matches "a red bus" better
    assert select_by_clip(img, candidates, "a red bus") == left


@pytest.mark.self_hosted
def test_clip_text_embedding_is_cached() -> None:
    """A repeated phrase reuses its cached text embedding (no re-encode)."""
    from unittest import mock

    from dimos.navigation.visual import grounding as g

    img = Image.from_numpy(np.zeros((32, 32, 3), dtype=np.uint8), format=ImageFormat.BGR)
    boxes = [(0.0, 0.0, 16.0, 16.0)]

    g._clip_text_cache.pop("a teal box", None)
    g.clip_scores(img, boxes, "a teal box")  # first call populates the cache
    assert "a teal box" in g._clip_text_cache

    # Second call with the same phrase must NOT touch the text encoder.
    model, _ = g._load_clip()

    def _fail_encode(*args, **kwargs):
        raise AssertionError("text encoder re-ran for a cached phrase")

    with mock.patch.object(model, "encode_text", _fail_encode):
        g.clip_scores(img, boxes, "a teal box")  # image encode only; text is cached


# --- Phase 6: natural-language parsing + unified resolver wiring ---


@pytest.mark.parametrize(
    "query,expected",
    [
        ("the leftmost chair", ("chair", "leftmost")),
        ("person on the right", ("person", "rightmost")),
        ("the biggest dog", ("dog", "largest")),
        ("central person", ("person", "center")),
        ("red mug", ("red mug", None)),
        ("person", ("person", None)),
        # Leading articles are stripped even with no spatial language — they
        # wreck YOLOE recall ("the person" finds 0 boxes, "person" finds 5).
        ("the person", ("person", None)),
        ("a bus", ("bus", None)),
        ("the red mug", ("red mug", None)),
        # Leading imperatives (anchored by an article) are stripped too.
        ("find the person", ("person", None)),
        ("go to the chair", ("chair", None)),
        ("navigate to the bus", ("bus", None)),
        ("pick up the bottle", ("bottle", None)),
        # No article anchor -> NOT treated as an imperative, left intact.
        ("pickup truck", ("pickup truck", None)),
    ],
)
def test_parse_grounding_query(query: str, expected: tuple) -> None:
    assert parse_grounding_query(query) == expected


def test_resolve_grounding_spatial_qualifier(image: Image) -> None:
    # Two chairs; the higher-confidence one is on the RIGHT, but "leftmost chair"
    # must return the left one — proving spatial parsing overrides raw confidence.
    detector = _FakeDetector(
        {
            "chair": [
                _FakeDetection("chair", 0.95, (100, 0, 120, 20)),  # right, top conf
                _FakeDetection("chair", 0.70, (0, 0, 20, 20)),  # left
            ]
        }
    )
    assert resolve_grounding(detector, image, "the leftmost chair") == (0.0, 0.0, 20.0, 20.0)


def test_resolve_grounding_spatial_plural_singularizes(image: Image) -> None:
    # "the leftmost chairs" (plural) must ground the "chair" class and pick the
    # left box — proving the main path singularizes, not just the fallback.
    detector = _FakeDetector(
        {
            "chair": [
                _FakeDetection("chair", 0.95, (100, 0, 120, 20)),  # right
                _FakeDetection("chair", 0.70, (0, 0, 20, 20)),  # left
            ]
        }
    )
    assert resolve_grounding(detector, image, "the leftmost chairs") == (0.0, 0.0, 20.0, 20.0)


def test_resolve_grounding_tracking_prefers_prev_box(image: Image) -> None:
    detector = _FakeDetector(
        {
            "person": [
                _FakeDetection("person", 0.95, (200, 200, 240, 260)),  # different instance
                _FakeDetection("person", 0.70, (10, 10, 30, 40)),  # near prev_box
            ]
        }
    )
    prev = (11.0, 11.0, 31.0, 41.0)
    # Tracking continuity beats raw confidence.
    assert resolve_grounding(detector, image, "person", prev_box=prev) == (10.0, 10.0, 30.0, 40.0)


def test_resolve_grounding_attribute_uses_clip(image: Image, monkeypatch) -> None:
    detector = _FakeDetector(
        {
            "red mug": [
                _FakeDetection("mug", 0.9, (0, 0, 10, 10)),
                _FakeDetection("mug", 0.8, (50, 50, 60, 60)),
            ]
        }
    )
    # Force CLIP to prefer the SECOND candidate regardless of confidence.
    monkeypatch.setattr(grounding_mod, "clip_scores", lambda img, cands, phrase: [0.1, 0.9])
    assert resolve_grounding(detector, image, "red mug") == (50.0, 50.0, 60.0, 60.0)


def test_resolve_grounding_attribute_recall_fallback_to_head_noun(image: Image) -> None:
    # Detector finds nothing for the full phrase but finds the head noun "mug".
    detector = _FakeDetector({"mug": [_FakeDetection("mug", 0.9, (1, 2, 3, 4))]})
    assert resolve_grounding(detector, image, "red mug") == (1.0, 2.0, 3.0, 4.0)


def test_resolve_grounding_recall_fallback_singularizes_plural_head_noun(image: Image) -> None:
    # "red mugs" finds nothing as a phrase; recovery grounds the singular "mug".
    detector = _FakeDetector({"mug": [_FakeDetection("mug", 0.9, (1, 2, 3, 4))]})
    assert resolve_grounding(detector, image, "red mugs") == (1.0, 2.0, 3.0, 4.0)


def test_resolve_grounding_plain_noun_returns_top_confidence(image: Image) -> None:
    detector = _FakeDetector(
        {
            "person": [
                _FakeDetection("person", 0.95, (5, 6, 7, 8)),
                _FakeDetection("person", 0.60, (0, 0, 1, 1)),
            ]
        }
    )
    assert resolve_grounding(detector, image, "person") == (5.0, 6.0, 7.0, 8.0)


def test_get_object_bbox_routes_spatial_query_without_vlm(image: Image) -> None:
    detector = _FakeDetector(
        {
            "person": [
                _FakeDetection("person", 0.95, (100, 0, 120, 20)),
                _FakeDetection("person", 0.70, (0, 0, 20, 20)),
            ]
        }
    )
    bbox = get_object_bbox(_RaisingVlModel(), image, "the leftmost person", detector=detector)
    assert bbox == (0.0, 0.0, 20.0, 20.0)  # geometric pick, VLM never touched


def test_get_object_bbox_passes_prev_box_for_tracking(image: Image) -> None:
    detector = _FakeDetector(
        {
            "person": [
                _FakeDetection("person", 0.95, (200, 200, 240, 260)),
                _FakeDetection("person", 0.70, (10, 10, 30, 40)),
            ]
        }
    )
    bbox = get_object_bbox(
        _RaisingVlModel(), image, "person", detector=detector, prev_box=(11.0, 11.0, 31.0, 41.0)
    )
    assert bbox == (10.0, 10.0, 30.0, 40.0)


def test_build_yoloe_grounding_detector_forwards_confidence(monkeypatch) -> None:
    import dimos.perception.detection.detectors.yoloe as yoloe_mod

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(yoloe_mod, "Yoloe2DDetector", _capture)
    det = build_yoloe_grounding_detector(confidence=0.25, iou_threshold=0.8)

    assert det is not None
    assert captured["confidence"] == 0.25  # recall knob threaded through
    assert captured["iou_threshold"] == 0.8  # NMS knob threaded through
    assert captured["max_area_ratio"] is None


# --- Ordinal spatial selectors ("second from the left") ---


@pytest.mark.parametrize(
    "query,expected",
    [
        ("the second chair from the left", ("chair", "from-left:2")),
        ("third person from the right", ("person", "from-right:3")),
        ("last car from the left", ("car", "from-left:last")),
        ("the 2nd dog from the top", ("dog", "from-top:2")),
    ],
)
def test_parse_grounding_query_ordinals(query: str, expected: tuple) -> None:
    assert parse_grounding_query(query) == expected


def test_select_by_position_ordinal_index() -> None:
    # Three boxes at x-centers 10, 20, 30 (given out of order to prove sorting).
    boxes = [(25.0, 0.0, 35.0, 10.0), (5.0, 0.0, 15.0, 10.0), (15.0, 0.0, 25.0, 10.0)]
    assert select_by_position(boxes, "from-left:1") == (5.0, 0.0, 15.0, 10.0)
    assert select_by_position(boxes, "from-left:2") == (15.0, 0.0, 25.0, 10.0)
    assert select_by_position(boxes, "from-right:1") == (25.0, 0.0, 35.0, 10.0)
    assert select_by_position(boxes, "from-left:last") == (25.0, 0.0, 35.0, 10.0)
    assert select_by_position(boxes, "from-left:5") is None  # out of range


def test_resolve_grounding_ordinal_end_to_end(image: Image) -> None:
    detector = _FakeDetector(
        {
            "person": [
                _FakeDetection("person", 0.9, (0, 0, 20, 20)),  # left
                _FakeDetection("person", 0.8, (100, 0, 120, 20)),  # right
                _FakeDetection("person", 0.7, (50, 0, 70, 20)),  # middle
            ]
        }
    )
    assert resolve_grounding(detector, image, "the second person from the left") == (
        50.0,
        0.0,
        70.0,
        20.0,
    )


# --- Real-model integration (self_hosted: deselected by the default marker) ---


class _TripwireVlModel:
    """Fails the test if the VLM is touched; supplies a known bbox if it is."""

    def __init__(self) -> None:
        self.called = False

    def query(self, image, query):
        self.called = True
        return '{"bbox": [1, 2, 3, 4]}'


def test_get_object_bboxes_returns_all_candidates_without_vlm(image: Image) -> None:
    detector = _FakeDetector(
        {
            "chair": [  # "all the chairs" -> singularized to the class "chair"
                _FakeDetection("chair", 0.9, (0, 0, 10, 10)),
                _FakeDetection("chair", 0.8, (20, 0, 30, 10)),
                _FakeDetection("chair", 0.7, (40, 0, 50, 10)),
            ]
        }
    )
    boxes = get_object_bboxes(_RaisingVlModel(), image, "all the chairs", detector=detector)
    assert boxes == [(0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0), (40.0, 0.0, 50.0, 10.0)]


def test_get_object_bboxes_falls_back_to_single_vlm_box(image: Image) -> None:
    detector = _FakeDetector({})  # YOLOE finds nothing
    vl = _StubVlModel('{"bbox": [1, 2, 3, 4]}')
    assert get_object_bboxes(vl, image, "chair", detector=detector) == [(1.0, 2.0, 3.0, 4.0)]
    assert vl.calls == 1


def test_get_object_bboxes_empty_when_nothing_found(image: Image) -> None:
    vl = _StubVlModel("no json here")  # extract_json -> None
    assert get_object_bboxes(vl, image, "chair") == []


@pytest.mark.parametrize(
    "plural,singular",
    [
        ("people", "person"),
        ("persons", "person"),
        ("children", "child"),
        ("men", "man"),
        ("chairs", "chair"),
        ("cars", "car"),
        ("buses", "bus"),
        ("boxes", "box"),
        ("berries", "berry"),
        # Non-plurals / already-singular must be left intact.
        ("bus", "bus"),
        ("gas", "gas"),
        ("lens", "lens"),
        ("glass", "glass"),
        ("person", "person"),
        ("dog", "dog"),
    ],
)
def test_singularize(plural: str, singular: str) -> None:
    assert singularize(plural) == singular


def test_get_object_bboxes_singularizes_plural_query(image: Image) -> None:
    # "all the people" must ground the class "person", not the literal "people".
    detector = _FakeDetector({"person": [_FakeDetection("person", 0.9, (1, 2, 3, 4))]})
    assert get_object_bboxes(_RaisingVlModel(), image, "all the people", detector=detector) == [
        (1.0, 2.0, 3.0, 4.0)
    ]


@pytest.mark.self_hosted
def test_real_yoloe_routing_on_bus_image() -> None:
    """End-to-end on bus.jpg with the real YOLOE model: spatial, ordinal, fallback."""
    from pathlib import Path

    import ultralytics

    img = Image.from_file(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    detector = build_yoloe_grounding_detector()
    assert detector is not None

    # Round to whole pixels: GPU/MPS inference is non-deterministic in the ~5th
    # decimal between calls, so compare boxes at pixel resolution, not exact float.
    def _px(b):
        return tuple(round(v) for v in b)

    persons = sorted(
        (_px(b) for b in ground_candidates_with_yoloe(detector, img, "person")),
        key=lambda b: (b[0] + b[2]) / 2,  # left-to-right by x-center
    )
    assert len(persons) >= 2  # bus.jpg has several people

    # Spatial + ordinal selectors pick the geometrically-correct instance.
    assert _px(resolve_grounding(detector, img, "the leftmost person")) == persons[0]
    assert _px(resolve_grounding(detector, img, "the second person from the left")) == persons[1]
    assert _px(resolve_grounding(detector, img, "the last person from the left")) == persons[-1]

    # Fast-path hit: the VLM tripwire must stay untouched.
    trip = _TripwireVlModel()
    assert _px(get_object_bbox(trip, img, "the leftmost person", detector=detector)) == persons[0]
    assert trip.called is False

    # Miss ("banana") falls back to the VLM, which supplies the box.
    trip2 = _TripwireVlModel()
    bbox = get_object_bbox(trip2, img, "banana", detector=detector)
    assert trip2.called is True
    assert bbox == (1.0, 2.0, 3.0, 4.0)

    detector.stop()


@pytest.mark.self_hosted
def test_yoloe_visual_prompt_path_and_mode_switch() -> None:
    """Visual (bbox) prompts run (previously crashed) and text<->bbox switches cleanly."""
    from pathlib import Path

    import ultralytics

    img = Image.from_file(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    detector = build_yoloe_grounding_detector(confidence=0.25)
    assert detector is not None

    ref = ground_candidates_with_yoloe(detector, img, "person")[0]

    # Visual (bbox) prompt: previously crashed in ultralytics NMS.
    detector.set_prompts(bboxes=np.array([list(ref)], dtype=np.float64))
    assert len(detector.process_image(img).detections) > 0

    # Switching back to text must work — the VP predictor used to corrupt the
    # model's class names (dict -> list), breaking the next text set_prompts.
    detector.set_prompts(text=["bus"])
    assert len(detector.process_image(img).detections) >= 1

    detector.stop()


@pytest.mark.self_hosted
def test_yoloe_grounds_grayscale_frame() -> None:
    """A single-channel (grayscale) frame is promoted to BGR, not crashed on."""
    from pathlib import Path

    import cv2
    import ultralytics

    bgr = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    gray = Image.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), format=ImageFormat.BGR)

    detector = build_yoloe_grounding_detector(confidence=0.25)
    assert detector is not None
    assert len(ground_candidates_with_yoloe(detector, gray, "person")) > 0
    detector.stop()


class _ConfFakeDetector:
    """Fake whose process_image only 'sees' the object below a confidence floor."""

    def __init__(self, present_below: float) -> None:
        self.confidence = 0.6
        self._prompt: str | None = None
        self.present_below = present_below

    def set_prompts(self, text: list[str] | None = None, bboxes=None) -> None:
        self._prompt = text[0] if text else None

    def process_image(self, image: Image) -> _FakeResult:
        if self.confidence <= self.present_below:
            return _FakeResult([_FakeDetection(self._prompt or "", 0.3, (1, 2, 3, 4))])
        return _FakeResult([])


def test_resolve_grounding_retries_at_lower_confidence(image: Image) -> None:
    # Object only detectable at conf <= 0.3; default 0.6 finds nothing, so the
    # recall retry should fire and find it.
    detector = _ConfFakeDetector(present_below=0.3)
    assert resolve_grounding(detector, image, "bottle") == (1.0, 2.0, 3.0, 4.0)
    assert detector.confidence == 0.6  # restored after the retry


def test_resolve_grounding_no_retry_when_found_at_default(image: Image) -> None:
    # Found at the default confidence -> no retry, confidence untouched.
    detector = _ConfFakeDetector(present_below=0.6)
    assert resolve_grounding(detector, image, "bottle") == (1.0, 2.0, 3.0, 4.0)
    assert detector.confidence == 0.6


def test_get_object_bboxes_retries_at_lower_confidence(image: Image) -> None:
    # Multi-object path also does the cheap recall retry before the VLM.
    detector = _ConfFakeDetector(present_below=0.3)
    boxes = get_object_bboxes(_RaisingVlModel(), image, "all the bottles", detector=detector)
    assert boxes == [(1.0, 2.0, 3.0, 4.0)]
    assert detector.confidence == 0.6  # restored


@pytest.mark.self_hosted
def test_clip_scores_handles_grayscale_image() -> None:
    """clip_scores promotes a grayscale frame to 3 channels instead of crashing."""
    from dimos.navigation.visual.grounding import clip_scores

    gray = Image.from_numpy(np.zeros((32, 32), dtype=np.uint8), format=ImageFormat.BGR)
    scores = clip_scores(gray, [(0.0, 0.0, 16.0, 16.0), (16.0, 16.0, 32.0, 32.0)], "a box")
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


# --- Relational grounding ("the cup next to the laptop") ---


@pytest.mark.parametrize(
    "query,expected",
    [
        ("the cup next to the laptop", ("the cup", "near", "the laptop")),
        ("bottle near the keyboard", ("bottle", "near", "the keyboard")),
        ("a chair beside the table", ("a chair", "near", "the table")),
        ("the bottle to the left of the laptop", ("the bottle", "left", "the laptop")),
        ("cup right of the plate", ("cup", "right", "the plate")),
        ("the book above the desk", ("the book", "above", "the desk")),
        ("a box under the chair", ("a box", "below", "the chair")),
        ("person", None),  # no relation
        ("the nearest person", None),  # "near" must not match inside "nearest"
    ],
)
def test_parse_relational_query(query: str, expected) -> None:
    assert parse_relational_query(query) == expected


def test_select_by_nearest_to_reference() -> None:
    reference = (100.0, 100.0, 140.0, 140.0)  # center (120, 120)
    near = (110.0, 110.0, 130.0, 130.0)  # center (120, 120)
    far = (0.0, 0.0, 20.0, 20.0)
    assert select_by_nearest_to_reference([far, near], reference) == near
    assert select_by_nearest_to_reference([], reference) is None


def test_select_by_relation_directional() -> None:
    reference = (100.0, 100.0, 140.0, 140.0)  # center (120, 120)
    left = (0.0, 100.0, 40.0, 140.0)  # center cx=20  (left of ref)
    right = (200.0, 100.0, 240.0, 140.0)  # center cx=220 (right of ref)
    above = (100.0, 0.0, 140.0, 40.0)  # center cy=20  (above ref)
    below = (100.0, 200.0, 140.0, 240.0)  # center cy=220 (below ref)
    pool = [left, right, above, below]
    assert select_by_relation(pool, "left", reference) == left
    assert select_by_relation(pool, "right", reference) == right
    assert select_by_relation(pool, "above", reference) == above
    assert select_by_relation(pool, "below", reference) == below
    # Nothing on the requested side -> None (caller falls to the VLM).
    assert select_by_relation([right], "left", reference) is None


def test_resolve_grounding_relational_picks_nearest(image: Image) -> None:
    detector = _FakeDetector(
        {
            "laptop": [_FakeDetection("laptop", 0.9, (100, 100, 140, 130))],
            "cup": [
                _FakeDetection("cup", 0.9, (0, 0, 20, 20)),  # far from laptop
                _FakeDetection("cup", 0.8, (110, 110, 130, 130)),  # next to laptop
            ],
        }
    )
    assert resolve_grounding(detector, image, "the cup next to the laptop") == (
        110.0,
        110.0,
        130.0,
        130.0,
    )


def test_resolve_grounding_relational_none_when_reference_missing(image: Image) -> None:
    # Object present but reference absent -> None (caller falls to the VLM).
    detector = _FakeDetector({"cup": [_FakeDetection("cup", 0.9, (0, 0, 20, 20))]})
    assert resolve_grounding(detector, image, "the cup next to the laptop") is None


def test_resolve_grounding_relational_directional_end_to_end(image: Image) -> None:
    detector = _FakeDetector(
        {
            "laptop": [_FakeDetection("laptop", 0.9, (100, 100, 140, 140))],
            "bottle": [
                _FakeDetection("bottle", 0.9, (0, 100, 40, 140)),  # left of laptop
                _FakeDetection("bottle", 0.8, (200, 100, 240, 140)),  # right of laptop
            ],
        }
    )
    assert resolve_grounding(detector, image, "the bottle to the left of the laptop") == (
        0.0,
        100.0,
        40.0,
        140.0,
    )
    assert resolve_grounding(detector, image, "the bottle right of the laptop") == (
        200.0,
        100.0,
        240.0,
        140.0,
    )


def test_resolve_grounding_relational_uses_recall_retry(image: Image) -> None:
    # "cup" only detectable below conf 0.3; relational grounding must still find
    # it via the recall retry instead of giving up to the VLM.
    class _ConfRelFake:
        def __init__(self) -> None:
            self.confidence = 0.6
            self._prompt: str | None = None

        def set_prompts(self, text: list[str] | None = None, bboxes=None) -> None:
            self._prompt = text[0] if text else None

        def process_image(self, image: Image) -> _FakeResult:
            if self._prompt == "laptop":
                return _FakeResult([_FakeDetection("laptop", 0.9, (100, 100, 140, 140))])
            if self._prompt == "cup" and self.confidence <= 0.3:
                return _FakeResult([_FakeDetection("cup", 0.3, (110, 110, 130, 130))])
            return _FakeResult([])

    detector = _ConfRelFake()
    assert resolve_grounding(detector, image, "the cup next to the laptop") == (
        110.0,
        110.0,
        130.0,
        130.0,
    )
    assert detector.confidence == 0.6  # restored
