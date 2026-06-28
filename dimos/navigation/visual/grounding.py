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

from typing import Any, Callable

import numpy as np

from dimos.models.qwen.bbox import BBox
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Lazily-loaded, process-wide CLIP cache (see :func:`_load_clip`). A CLIP load is
# seconds of work and the weights are hundreds of MB, so the model and its
# preprocessing transform are built once on first use and reused thereafter.
_clip_model: Any = None
_clip_preprocess: Callable[..., Any] | None = None


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


def ground_candidates_with_yoloe(detector, image, description: str) -> list[BBox]:
    """Return all YOLOE match bboxes for `description`, highest-confidence first.

    The multi-candidate counterpart to :func:`ground_with_yoloe`: where that
    returns only the single best box, this returns every detection so callers can
    disambiguate among them (e.g. "the leftmost chair") via
    :func:`select_by_position`.

    Args:
        detector: A constructed open-vocab detector (e.g. ``Yoloe2DDetector`` in
            ``YoloePromptMode.PROMPT``) exposing ``set_prompts`` and
            ``process_image``.
        image: A ``dimos.msgs.sensor_msgs.Image`` to ground against.
        description: Natural-language name of the object to locate.

    Returns:
        Every matching detection's ``(x1, y1, x2, y2)`` bbox as a tuple of
        floats, sorted by detection confidence in descending order. Empty list
        if nothing matched.
    """
    # Skip the costly text re-encode when the target hasn't changed since this
    # detector's last call (the steady-state per-frame tracking path).
    if getattr(detector, "_yoloe_grounder_prompt", None) != description:
        detector.set_prompts(text=[description])
        detector._yoloe_grounder_prompt = description

    result = detector.process_image(image)

    detections = sorted(result.detections, key=lambda d: d.confidence, reverse=True)
    return [
        (float(x1), float(y1), float(x2), float(y2))
        for x1, y1, x2, y2 in (d.bbox for d in detections)
    ]


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
    candidates = ground_candidates_with_yoloe(detector, image, description)
    return candidates[0] if candidates else None


def select_by_position(candidates: list[BBox], qualifier: str) -> BBox | None:
    """Pick one bbox from `candidates` by a geometric `qualifier`, model-free.

    Disambiguates a set of same-class boxes (as returned by
    :func:`ground_candidates_with_yoloe`) using box geometry alone — no model,
    no image content. Useful for resolving spatial language like "the leftmost
    chair" or "the largest screen".

    Args:
        candidates: Bounding boxes to choose among, conventionally ordered by
            descending confidence (the order :func:`ground_candidates_with_yoloe`
            produces). Ties resolve to the first box in this order, so a
            confidence-sorted input makes ties deterministic and sensible.
        qualifier: Case-insensitive spatial selector. One of ``leftmost``,
            ``rightmost``, ``topmost``, ``bottommost``, ``largest``,
            ``smallest``, or ``center`` (the box whose center is closest to the
            mean of all candidate centers).

    Returns:
        The chosen ``(x1, y1, x2, y2)`` bbox, or ``None`` if ``candidates`` is
        empty.

    Raises:
        ValueError: If ``qualifier`` is not a recognized selector.
    """
    if not candidates:
        return None

    def _cx(b: BBox) -> float:
        return (b[0] + b[2]) / 2.0

    def _cy(b: BBox) -> float:
        return (b[1] + b[3]) / 2.0

    def _area(b: BBox) -> float:
        return abs(b[2] - b[0]) * abs(b[3] - b[1])

    key = qualifier.strip().lower()
    if key == "leftmost":
        return min(candidates, key=_cx)
    if key == "rightmost":
        return max(candidates, key=_cx)
    if key == "topmost":
        return min(candidates, key=_cy)
    if key == "bottommost":
        return max(candidates, key=_cy)
    if key == "largest":
        return max(candidates, key=_area)
    if key == "smallest":
        return min(candidates, key=_area)
    if key == "center":
        mean_cx = sum(_cx(b) for b in candidates) / len(candidates)
        mean_cy = sum(_cy(b) for b in candidates) / len(candidates)
        return min(
            candidates,
            key=lambda b: (_cx(b) - mean_cx) ** 2 + (_cy(b) - mean_cy) ** 2,
        )

    raise ValueError(f"Unknown position qualifier: {qualifier!r}")


def ground_with_position(
    detector, image, description: str, qualifier: str | None = None
) -> BBox | None:
    """Ground `description` and optionally resolve it spatially via `qualifier`.

    A thin convenience over :func:`ground_candidates_with_yoloe`: with a
    ``qualifier`` it returns the geometrically-selected box (see
    :func:`select_by_position`); without one it returns the top-confidence box,
    matching :func:`ground_with_yoloe`.

    Args:
        detector: A constructed open-vocab detector.
        image: A ``dimos.msgs.sensor_msgs.Image`` to ground against.
        description: Natural-language name of the object to locate.
        qualifier: Optional spatial selector passed to
            :func:`select_by_position`.

    Returns:
        The chosen ``(x1, y1, x2, y2)`` bbox, or ``None`` if nothing matched.
    """
    candidates = ground_candidates_with_yoloe(detector, image, description)
    if qualifier is None:
        return candidates[0] if candidates else None
    return select_by_position(candidates, qualifier)


def _load_clip() -> tuple[Any, Callable[..., Any]]:
    """Lazily build and cache the CLIP ViT-B/32 model + preprocess on CPU.

    The model is constructed on first use and memoized at module level so every
    subsequent re-rank reuses it — a CLIP load is seconds of work and the weights
    are hundreds of MB. Kept on CPU deliberately: re-ranking a handful of crops is
    cheap and the fast path must not contend for the GPU the detector may hold.

    Raises:
        RuntimeError: If the ``clip`` package or ``torch`` cannot be imported, so
            the resilient router can fall back to the geometry- or VLM-only paths
            instead of crashing.
    """
    global _clip_model, _clip_preprocess
    if _clip_model is None or _clip_preprocess is None:
        try:
            import clip
            import torch
        except ImportError as e:
            raise RuntimeError(
                "CLIP attribute re-ranking needs the `clip` package and torch, "
                "neither of which is importable in this deployment."
            ) from e

        model, preprocess = clip.load("ViT-B/32", device="cpu")
        model.eval()
        torch.set_grad_enabled(False)
        _clip_model, _clip_preprocess = model, preprocess
    return _clip_model, _clip_preprocess


def clip_scores(image, candidates: list[BBox], phrase: str) -> list[float]:
    """CLIP cosine similarity of each candidate's crop to `phrase`.

    For every candidate box this crops that region out of the image, encodes the
    crop and the text `phrase` with CLIP, and returns their cosine similarity —
    one float per candidate, in the same order as ``candidates``. This is what
    lets re-ranking honour attributive language ("the red mug") that the detector
    and geometry alone can't resolve.

    The CLIP model is loaded lazily and cached at module level (see
    :func:`_load_clip`), so only the first call pays the load cost. Runs on CPU.

    Args:
        image: The ``dimos.msgs.sensor_msgs.Image`` the boxes were grounded in;
            ``image.to_opencv()`` supplies the BGR pixels, converted to RGB here.
        candidates: ``(x1, y1, x2, y2)`` boxes to score, in pixel coordinates.
        phrase: The attributive phrase to match each crop against, e.g. "red mug".

    Returns:
        One cosine-similarity float per candidate, in ``candidates`` order. Empty
        list if ``candidates`` is empty.

    Raises:
        RuntimeError: If CLIP/torch are unavailable (propagated from
            :func:`_load_clip`).
    """
    if not candidates:
        return []

    import clip
    import torch
    from PIL import Image as PILImage

    model, preprocess = _load_clip()

    # BGR -> RGB; a contiguous copy keeps PIL happy with the reversed-stride view.
    rgb = np.ascontiguousarray(image.to_opencv()[:, :, ::-1])
    height, width = rgb.shape[:2]

    crops = []
    for x1, y1, x2, y2 in candidates:
        # Clamp to the frame and guard degenerate/empty boxes: a zero-area crop
        # would make PIL choke, so such a box falls back to the whole frame (it
        # simply scores uninformatively rather than crashing the batch).
        ix1 = max(0, min(int(round(x1)), width - 1))
        iy1 = max(0, min(int(round(y1)), height - 1))
        ix2 = max(ix1 + 1, min(int(round(x2)), width))
        iy2 = max(iy1 + 1, min(int(round(y2)), height))
        crop = rgb[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            crop = rgb
        crops.append(preprocess(PILImage.fromarray(crop)))

    batch = torch.stack(crops)
    text = clip.tokenize([phrase])

    with torch.no_grad():
        image_features = model.encode_image(batch)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # (N, D) @ (D, 1) -> (N, 1) cosine similarities, one row per crop.
        sims = (image_features @ text_features.T).squeeze(-1)

    return [float(s) for s in sims]


def select_by_clip(
    image,
    candidates: list[BBox],
    phrase: str,
    *,
    scorer: Callable[[Any, list[BBox], str], list[float]] | None = None,
) -> BBox | None:
    """Pick the candidate whose crop best matches `phrase`, by CLIP similarity.

    The attributive counterpart to :func:`select_by_position`: where that
    disambiguates same-class boxes by geometry, this disambiguates them by
    appearance — returning the box whose image content scores highest against
    `phrase`. Use it for queries geometry can't answer, like "the red mug" among
    several mugs or "the person in a blue shirt".

    Args:
        image: The ``dimos.msgs.sensor_msgs.Image`` the boxes were grounded in.
        candidates: ``(x1, y1, x2, y2)`` boxes to choose among.
        phrase: The attributive phrase to match against, e.g. "red mug".
        scorer: Callable ``(image, candidates, phrase) -> list[float]`` returning
            one score per candidate; defaults to :func:`clip_scores`. Injectable
            so unit tests can pass a fake scorer and avoid loading CLIP.

    Returns:
        The highest-scoring ``(x1, y1, x2, y2)`` box, or ``None`` if
        ``candidates`` is empty. Ties resolve to the earliest such box in
        ``candidates`` order.
    """
    if not candidates:
        return None
    score = scorer if scorer is not None else clip_scores
    scores = score(image, candidates, phrase)
    best = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best]


def ground_with_attribute(
    detector, image, object_noun: str, phrase: str | None = None
) -> BBox | None:
    """Ground `object_noun` and optionally re-rank its matches by `phrase`.

    A thin convenience over :func:`ground_candidates_with_yoloe`: it grounds the
    bare object class (e.g. "mug"), then with an attributive ``phrase`` returns
    the crop that best matches it via :func:`select_by_clip` ("the red mug");
    without one it returns the top-confidence box, matching
    :func:`ground_with_yoloe`. Detecting the plain noun and re-ranking by
    appearance beats prompting the detector with the full phrase, which open-vocab
    detectors handle poorly.

    Args:
        detector: A constructed open-vocab detector.
        image: A ``dimos.msgs.sensor_msgs.Image`` to ground against.
        object_noun: The bare object class to detect, e.g. "mug".
        phrase: Optional attributive phrase to re-rank candidates by, e.g.
            "red mug".

    Returns:
        The chosen ``(x1, y1, x2, y2)`` bbox, or ``None`` if nothing matched.
    """
    candidates = ground_candidates_with_yoloe(detector, image, object_noun)
    if phrase is None:
        return candidates[0] if candidates else None
    return select_by_clip(image, candidates, phrase)
