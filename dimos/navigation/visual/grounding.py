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

"""Open-vocabulary YOLOE grounding — the fast path for visual object grounding.

`ground_with_yoloe` turns a natural-language object description into a single
bounding box using the open-vocab YOLOE detector. It is the fast counterpart to
the Qwen-VLM grounding in ``query.get_object_bbox_from_image`` and returns the
same ``(x1, y1, x2, y2)`` float-tuple convention, so the two are interchangeable
behind ``query.get_object_bbox``.

The function takes an already-constructed detector and image — it does NOT build
them itself. It memoizes the last text prompt per detector: re-encoding the
prompt (a CLIP text-encoder pass) dominates per-call latency, so when the
description is unchanged from the detector's previous call — the common
navigation case of grounding the same object across consecutive frames — that
cost is skipped and only the detection runs.
"""

from __future__ import annotations

from typing import Any

from dimos.models.qwen.bbox import BBox
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def build_yoloe_grounding_detector() -> Any | None:
    """Best-effort construct the YOLOE fast-path detector; ``None`` on any failure.

    Grounding consumers use this to opt into the fast path without risking a
    crash: if YOLOE — or its weights or text encoder — is unavailable in the
    deployment, this returns ``None`` and the caller transparently falls back to
    the VLM-only path. ``max_area_ratio=None`` keeps large objects (e.g. a close
    person, a bus) instead of dropping anything over 30% of the frame.
    """
    try:
        from dimos.perception.detection.detectors.yoloe import (
            Yoloe2DDetector,
            YoloePromptMode,
        )

        return Yoloe2DDetector(prompt_mode=YoloePromptMode.PROMPT, max_area_ratio=None)
    except Exception:
        logger.warning(
            "YOLOE grounding detector unavailable; using VLM-only grounding.",
            exc_info=True,
        )
        return None


def ground_with_yoloe(detector, image, description: str) -> BBox | None:
    """Return the (x1,y1,x2,y2) bbox of the best YOLOE match for `description`, or None.

    Args:
        detector: A constructed open-vocab detector (e.g. ``Yoloe2DDetector`` in
            ``YoloePromptMode.PROMPT``) exposing ``set_prompts`` and
            ``process_image``.
        image: A ``dimos.msgs.sensor_msgs.Image`` to ground against.
        description: Natural-language name of the object to locate.

    Returns:
        The bounding box of the single highest-confidence detection as a
        ``(x1, y1, x2, y2)`` tuple of floats, or ``None`` if nothing matched.
    """
    # Skip the costly text re-encode when the target hasn't changed since this
    # detector's last call (the steady-state per-frame tracking path).
    if getattr(detector, "_yoloe_grounder_prompt", None) != description:
        detector.set_prompts(text=[description])
        detector._yoloe_grounder_prompt = description

    result = detector.process_image(image)

    detections = result.detections
    if not detections:
        return None

    best = max(detections, key=lambda d: d.confidence)
    x1, y1, x2, y2 = best.bbox
    return (float(x1), float(y1), float(x2), float(y2))
