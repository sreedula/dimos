"""Headless self-test for the YOLOE fast-path in query.get_object_bbox.

Proves both routing branches with stub VL models (no real Qwen call):
  Case A — fast-path hit: detector finds "person", so the VLM .query is NEVER
           called (the stub raises if it is).
  Case B — fallback: detector returns None for "banana", so get_object_bbox
           falls through to the VLM path, which IS called and supplies the bbox.
"""

from pathlib import Path

import ultralytics

from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.visual.query import get_object_bbox
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector, YoloePromptMode


class RaisingVlModel:
    """Stub VL model that fails loudly if its query path is ever taken."""

    def query(self, image, prompt):  # noqa: ANN001
        raise AssertionError("VLM .query was called — fast path should have handled this")


class StubVlModel:
    """Stub VL model that records the call and returns a known dummy bbox."""

    def __init__(self):
        self.called = False

    def query(self, image, prompt):  # noqa: ANN001
        self.called = True
        return '{"name": "banana", "bbox": [1, 2, 3, 4]}'


def main() -> int:
    sample_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    img = Image.from_file(str(sample_path))
    detector = Yoloe2DDetector(prompt_mode=YoloePromptMode.PROMPT, max_area_ratio=None)

    failures = 0

    # ---- Case A: fast-path hit, VLM must NOT be called ----
    try:
        bbox = get_object_bbox(RaisingVlModel(), img, "person", detector=detector)
        assert bbox is not None, "expected a bbox from the YOLOE fast path"
        assert len(bbox) == 4, f"expected a 4-tuple, got {bbox!r}"
        print(f"Case A (fast path hit, VLM not called): PASS  bbox={tuple(round(v) for v in bbox)}")
    except Exception as e:  # AssertionError from stub means the VLM was wrongly called
        failures += 1
        print(f"Case A: FAIL — {type(e).__name__}: {e}")

    # ---- Case B: no YOLOE match, must fall back to the VLM ----
    try:
        stub = StubVlModel()
        bbox = get_object_bbox(stub, img, "banana", detector=detector)
        assert stub.called, "VLM .query was not called on fallback"
        assert bbox == (1.0, 2.0, 3.0, 4.0), f"expected fallback bbox (1,2,3,4), got {bbox!r}"
        print(f"Case B (fallback to VLM, VLM called): PASS  bbox={bbox}")
    except Exception as e:
        failures += 1
        print(f"Case B: FAIL — {type(e).__name__}: {e}")

    detector.stop()

    print(f"\nRESULT: {'PASS' if failures == 0 else 'FAIL'} ({2 - failures}/2 cases passed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
