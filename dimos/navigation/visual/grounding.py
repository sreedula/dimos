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

from collections.abc import Callable
import re
from typing import Any

import numpy as np

from dimos.models.qwen.bbox import BBox
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Lazily-loaded, process-wide CLIP cache (see :func:`_load_clip`). A CLIP load is
# seconds of work and the weights are hundreds of MB, so the model and its
# preprocessing transform are built once on first use and reused thereafter.
_clip_model: Any = None
_clip_preprocess: Callable[..., Any] | None = None
# Normalized CLIP text embeddings, cached per phrase. A phrase's embedding is
# constant, so grounding the same attribute across frames ("the red mug") skips
# the redundant text-encoder pass after the first call.
_clip_text_cache: dict[str, Any] = {}


def build_yoloe_grounding_detector(
    confidence: float = 0.6, iou_threshold: float = 0.6
) -> Any | None:
    """Best-effort construct the YOLOE fast-path detector; ``None`` on any failure.

    Grounding consumers use this to opt into the fast path without risking a
    crash: if YOLOE — or its weights or text encoder — is unavailable in the
    deployment, this returns ``None`` and the caller transparently falls back to
    the VLM-only path. ``max_area_ratio=None`` keeps large objects (e.g. a close
    person, a bus) instead of dropping anything over 30% of the frame.

    Args:
        confidence: Minimum detection confidence (0-1]. The default 0.6 favors
            precision; lower it (e.g. 0.25) to raise recall on smaller or
            partially-occluded targets.
        iou_threshold: NMS IoU threshold (0-1]; raise it to keep more overlapping
            boxes in crowded scenes.
    """
    try:
        from dimos.perception.detection.detectors.yoloe import (
            Yoloe2DDetector,
            YoloePromptMode,
        )

        return Yoloe2DDetector(
            prompt_mode=YoloePromptMode.PROMPT,
            max_area_ratio=None,
            confidence=confidence,
            iou_threshold=iou_threshold,
        )
    except Exception:
        logger.warning(
            "YOLOE grounding detector unavailable; using VLM-only grounding.",
            exc_info=True,
        )
        return None


def ground_candidates_with_yoloe(
    detector, image, description: str, confidence: float | None = None
) -> list[BBox]:
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
    # detector's last call (the steady-state per-frame tracking path). But always
    # re-set if the detector was switched to a visual (bbox) prompt elsewhere,
    # else the stale memo would let process_image run with the old visual prompt
    # instead of grounding `description`.
    in_visual_mode = getattr(detector, "_visual_prompts", None) is not None
    if in_visual_mode or getattr(detector, "_yoloe_grounder_prompt", None) != description:
        detector.set_prompts(text=[description])
        detector._yoloe_grounder_prompt = description

    # Pass the per-call confidence only when overriding, so detectors whose
    # process_image takes no confidence argument keep working.
    if confidence is None:
        result = detector.process_image(image)
    else:
        result = detector.process_image(image, confidence=confidence)

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

    # Ordinal qualifiers like "from-left:2" (second from the left) or
    # "from-bottom:last": order the candidates along the axis and index in.
    ordinal = re.fullmatch(r"from-(left|right|top|bottom):(\d+|last)", key)
    if ordinal:
        direction, n = ordinal.group(1), ordinal.group(2)
        axis = _cx if direction in ("left", "right") else _cy
        reverse = direction in ("right", "bottom")
        ordered = sorted(candidates, key=axis, reverse=reverse)
        if n == "last":
            return ordered[-1]
        index = int(n) - 1
        return ordered[index] if 0 <= index < len(ordered) else None

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
    from PIL import Image as PILImage
    import torch

    model, preprocess = _load_clip()

    # Promote a grayscale (2D / single-channel) frame to 3 channels so the
    # BGR->RGB reversal and PIL conversion below don't choke on it.
    bgr = image.to_opencv()
    if bgr.ndim == 2:
        bgr = np.repeat(bgr[:, :, None], 3, axis=2)
    elif bgr.ndim == 3 and bgr.shape[2] == 1:
        bgr = np.repeat(bgr, 3, axis=2)

    # BGR -> RGB; a contiguous copy keeps PIL happy with the reversed-stride view.
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    height, width = rgb.shape[:2]

    crops = []
    for x1, y1, x2, y2 in candidates:
        # Clamp to the frame and guard degenerate/empty boxes: a zero-area crop
        # would make PIL choke, so such a box falls back to the whole frame (it
        # simply scores uninformatively rather than crashing the batch).
        ix1 = max(0, min(round(x1), width - 1))
        iy1 = max(0, min(round(y1), height - 1))
        ix2 = max(ix1 + 1, min(round(x2), width))
        iy2 = max(iy1 + 1, min(round(y2), height))
        crop = rgb[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            crop = rgb
        crops.append(preprocess(PILImage.fromarray(crop)))

    batch = torch.stack(crops)

    # Text embedding is constant per phrase — encode once, then reuse.
    text_features = _clip_text_cache.get(phrase)
    if text_features is None:
        with torch.no_grad():
            encoded = model.encode_text(clip.tokenize([phrase]))
            text_features = encoded / encoded.norm(dim=-1, keepdim=True)
        _clip_text_cache[phrase] = text_features

    with torch.no_grad():
        image_features = model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
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


def box_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes, in ``0..1``.

    The standard overlap metric: the area of the boxes' intersection divided by
    the area of their union. ``1.0`` for identical boxes, ``0.0`` when they don't
    touch. Used by :func:`select_nearest` to keep a tracked target locked onto the
    same instance across frames.

    Args:
        a: A ``(x1, y1, x2, y2)`` box.
        b: A ``(x1, y1, x2, y2)`` box.

    Returns:
        Their IoU as a float in ``[0.0, 1.0]``; ``0.0`` if they don't overlap or
        either box has zero area.
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = abs(a[2] - a[0]) * abs(a[3] - a[1])
    area_b = abs(b[2] - b[0]) * abs(b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def select_nearest(candidates: list[BBox], prev_box: BBox, *, min_iou: float = 0.0) -> BBox | None:
    """Pick the candidate most consistent with `prev_box`, for frame-to-frame tracking.

    The tracking counterpart to :func:`select_by_position` and
    :func:`select_by_clip`: where those disambiguate same-class boxes by geometry
    or appearance, this disambiguates by *continuity* — returning the box that
    best matches where the target was last seen. This stops the grounded box from
    flipping between instances (e.g. between two people) when confidence alone
    would jump frame to frame.

    Candidates are ranked primarily by highest IoU with ``prev_box``. When every
    candidate's IoU is ``0.0`` — the target briefly stopped overlapping its last
    position — it falls back to the smallest center-to-center distance, so the
    track survives short gaps instead of being dropped.

    Args:
        candidates: ``(x1, y1, x2, y2)`` boxes to choose among, conventionally
            ordered by descending confidence. Ties resolve to the earliest box in
            this order.
        prev_box: The target's last known ``(x1, y1, x2, y2)`` box.
        min_iou: If positive and the best candidate's IoU with ``prev_box`` is
            below it, return ``None`` — the target is considered lost rather than
            silently re-locked onto a non-overlapping box.

    Returns:
        The chosen ``(x1, y1, x2, y2)`` box, ``None`` if ``candidates`` is empty,
        or ``None`` if ``min_iou`` is positive and no candidate overlaps
        ``prev_box`` by at least that much.
    """
    if not candidates:
        return None

    ious = [box_iou(c, prev_box) for c in candidates]
    best_iou = max(ious)

    if best_iou > 0.0:
        if min_iou > 0.0 and best_iou < min_iou:
            return None
        return candidates[max(range(len(candidates)), key=lambda i: ious[i])]

    # No candidate overlaps the last box: a strict min_iou treats the target as
    # lost; otherwise fall back to the nearest center to track through the gap.
    if min_iou > 0.0:
        return None

    pcx = (prev_box[0] + prev_box[2]) / 2.0
    pcy = (prev_box[1] + prev_box[3]) / 2.0

    def _dist2(b: BBox) -> float:
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        return (cx - pcx) ** 2 + (cy - pcy) ** 2

    return min(candidates, key=_dist2)


def ground_with_tracking(
    detector,
    image,
    description: str,
    prev_box: BBox | None = None,
    *,
    min_iou: float = 0.0,
) -> BBox | None:
    """Ground `description`, preferring the candidate nearest `prev_box` when tracking.

    A thin convenience over :func:`ground_candidates_with_yoloe`: with a
    ``prev_box`` it returns the box most consistent with the target's last known
    position (see :func:`select_nearest`), keeping a track locked onto the same
    instance; without one it returns the top-confidence box, matching
    :func:`ground_with_yoloe`.

    Args:
        detector: A constructed open-vocab detector.
        image: A ``dimos.msgs.sensor_msgs.Image`` to ground against.
        description: Natural-language name of the object to locate.
        prev_box: The target's last known ``(x1, y1, x2, y2)`` box, or ``None`` on
            the first frame / when not tracking.
        min_iou: Forwarded to :func:`select_nearest`; if positive and no candidate
            overlaps ``prev_box`` by at least this much, the target is treated as
            lost and ``None`` is returned.

    Returns:
        The chosen ``(x1, y1, x2, y2)`` bbox, or ``None`` if nothing matched (or
        the target was lost under ``min_iou``).
    """
    candidates = ground_candidates_with_yoloe(detector, image, description)
    if prev_box is None:
        return candidates[0] if candidates else None
    return select_nearest(candidates, prev_box, min_iou=min_iou)


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


# Spatial language → canonical qualifier understood by ``select_by_position``.
# Multi-word phrases are listed before their single-word forms so they match
# first (e.g. "on the left" before bare "left").
_SPATIAL_PATTERNS: list[tuple[tuple[str, ...], str]] = [
    (("leftmost", "left-most", "on the left", "to the left", "left"), "leftmost"),
    (("rightmost", "right-most", "on the right", "to the right", "right"), "rightmost"),
    (("topmost", "top-most", "at the top", "top", "upper"), "topmost"),
    (("bottommost", "bottom-most", "at the bottom", "bottom", "lower"), "bottommost"),
    (("largest", "biggest", "nearest", "closest"), "largest"),
    (("smallest", "farthest", "furthest"), "smallest"),
    (("centermost", "center", "centre", "central", "middle", "in the middle"), "center"),
]

# Irregular plurals worth handling for open-vocab grounding. Includes -ves and
# -oes forms whose singular the regular -s rule gets wrong (it would strip only
# the "s", yielding non-words like "knive"/"tomatoe" that ground nothing). The
# -ves -> -f vs -fe split (leaf vs knife) is irregular, so these are explicit.
_IRREGULAR_PLURALS: dict[str, str] = {
    "people": "person",
    "persons": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "feet": "foot",
    "teeth": "tooth",
    "geese": "goose",
    "mice": "mouse",
    "knives": "knife",
    "lives": "life",
    "wives": "wife",
    "leaves": "leaf",
    "shelves": "shelf",
    "wolves": "wolf",
    "halves": "half",
    "calves": "calf",
    "loaves": "loaf",
    "thieves": "thief",
    "scarves": "scarf",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "heroes": "hero",
    "echoes": "echo",
}
# Non-plural words ending in "s" that must NOT be naively singularized.
_KEEP_AS_IS: set[str] = {
    "lens",
    "iris",
    "octopus",
    "scissors",
    "glasses",
    "pants",
    "series",
    "species",
    "news",
    "compass",
    "grass",
    "glass",
}


def singularize(word: str) -> str:
    """Best-effort singular of a noun for open-vocab grounding ("chairs" -> "chair").

    Handles common irregular plurals and regular ``-ies``/``-es``/``-s`` endings,
    and leaves known non-plural ``-s`` words ("bus", "lens") alone. Imperfect by
    design: a wrong guess simply grounds nothing and falls back, so it only ever
    helps recall — it never makes a correct query worse.
    """
    lower = word.lower()
    if lower in _KEEP_AS_IS:
        return word
    if lower in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[lower]
    if len(lower) > 3 and lower.endswith("ies"):
        return word[:-3] + "y"
    if len(lower) > 4 and lower.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    # >3 guard keeps short non-plurals like "bus"/"gas" intact.
    if len(lower) > 3 and lower.endswith("s") and not lower.endswith("ss"):
        return word[:-1]
    return word


# Leading quantifiers/determiners to strip for multi-object grounding ("count the
# X" / "all the X"), where the count is implicit (every instance is returned).
# Longest-first so "all the" beats "all" and "a couple of" beats "a".
_QUANTIFIERS: tuple[str, ...] = (
    "all of the", "all the", "all", "both of the", "both", "every", "each",
    "a couple of", "a couple", "a few of", "a few", "a number of", "a bunch of",
    "a lot of", "lots of", "several", "many", "some of the", "some",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "the", "a", "an",
)
_QUANTIFIER_RE = re.compile(
    r"^\s*(?:\d+|"
    + "|".join(re.escape(q) for q in sorted(_QUANTIFIERS, key=len, reverse=True))
    + r")\s+",
    re.IGNORECASE,
)


def strip_quantifiers(phrase: str) -> str:
    """Remove leading quantifiers/numbers ("both", "the three", "several") from a
    multi-object phrase, leaving the bare class ("both chairs" -> "chairs").

    Applied repeatedly so stacked determiners collapse ("the three people" ->
    "people"). Returns the phrase unchanged if it is entirely quantifiers.
    """
    out = phrase.strip()
    prev = None
    while prev != out and out:
        prev = out
        out = _QUANTIFIER_RE.sub("", out, count=1).strip()
    return out or phrase.strip()


# Ordinal words → 1-based rank ("last" handled specially by select_by_position).
_ORDINALS: dict[str, str] = {
    "first": "1",
    "1st": "1",
    "second": "2",
    "2nd": "2",
    "third": "3",
    "3rd": "3",
    "fourth": "4",
    "4th": "4",
    "fifth": "5",
    "5th": "5",
    "sixth": "6",
    "6th": "6",
    "last": "last",
}
# "[the] <ordinal> <noun> from the <left|right|top|bottom>" — the noun sits
# between the ordinal and the direction (e.g. "the second chair from the left").
_ORDINAL_RE = re.compile(
    r"\b(?:the\s+)?(" + "|".join(_ORDINALS) + r")\s+(.+?)\s+from\s+the\s+(left|right|top|bottom)\b",
    re.IGNORECASE,
)


def parse_grounding_query(description: str) -> tuple[str, str | None]:
    """Split a query into (object phrase, spatial qualifier or None).

    Recognizes spatial language like "the leftmost chair" or "person on the
    right" and returns the canonical qualifier for :func:`select_by_position`
    plus the remaining object phrase with the spatial words and any leading
    article removed. With no spatial language, returns ``(cleaned text, None)``.

    Args:
        description: The natural-language grounding query.

    Returns:
        ``(object_phrase, qualifier)`` where ``qualifier`` is one of the
        selectors understood by :func:`select_by_position`, or ``None``.
    """
    # Normalize whitespace and drop surrounding punctuation an agent or user may
    # add ("the person." / "the chair?"), which would otherwise be grounded
    # literally ("person.") and find nothing.
    text = " ".join(description.strip().split()).strip(" ,.!?;:")

    # Strip a leading imperative that agents prepend ("find the person", "go to
    # the chair"). Only when followed by an article, which anchors that the rest
    # is the object reference — so e.g. "pickup truck" (no article) is untouched.
    text = re.sub(
        r"^(?:please\s+)?(?:find|locate|detect|go to|navigate to|show me|get|"
        r"grab|bring me|pick up)\s+(?=(?:the|a|an)\s+)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Ordinals first ("the second chair from the left"), so the bare-direction
    # patterns below don't capture the "left" inside them.
    ordinal_match = _ORDINAL_RE.search(text)
    if ordinal_match:
        rank = _ORDINALS[ordinal_match.group(1).lower()]
        noun = re.sub(r"^\s*(the|a|an)\s+", "", ordinal_match.group(2), flags=re.IGNORECASE)
        noun = " ".join(noun.split()).strip(" ,.")
        return (noun or text), f"from-{ordinal_match.group(3).lower()}:{rank}"

    for phrases, qualifier in _SPATIAL_PATTERNS:
        # Longest phrase first so a shorter form doesn't match inside a longer one
        # and leave a fragment ("in the middle" must win over "middle", else the
        # object becomes "person in the").
        for phrase in sorted(phrases, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            if pattern.search(text):
                stripped = pattern.sub(" ", text)
                stripped = re.sub(r"^\s*(the|a|an)\s+", " ", stripped, flags=re.IGNORECASE)
                stripped = " ".join(stripped.split()).strip(" ,.")
                return (stripped or text), qualifier

    # No spatial language: still strip a leading article. YOLOE's text encoder is
    # sensitive to it — e.g. "person" finds 5 boxes but "the person" finds 0.
    plain = re.sub(r"^\s*(the|a|an)\s+", "", text, flags=re.IGNORECASE).strip()
    return (plain or text), None


def _is_attributive(object_phrase: str) -> bool:
    """Heuristic: a multi-word phrase likely carries an appearance attribute."""
    return len(object_phrase.split()) > 1


def _singularize_head(phrase: str) -> str:
    """Singularize the last word of a phrase ("red chairs" -> "red chair")."""
    words = phrase.split()
    if not words:
        return phrase
    words[-1] = singularize(words[-1])
    return " ".join(words)


# Connector phrase -> canonical relation. Spaces around each form (in the regex)
# keep them from matching inside words (e.g. "near" won't match "nearest").
_RELATIONS: dict[str, str] = {
    "next to": "near",
    "closest to": "near",
    "close to": "near",
    "nearest to": "near",
    "beside": "near",
    "near": "near",
    "to the left of": "left",
    "left of": "left",
    "to the right of": "right",
    "right of": "right",
    "above": "above",
    "on top of": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
}
# Longest connectors first so "to the left of" matches before "left of"/"of".
_RELATION_RE = re.compile(
    r"^(.+?)\s+("
    + "|".join(re.escape(k) for k in sorted(_RELATIONS, key=len, reverse=True))
    + r")\s+(.+)$",
    re.IGNORECASE,
)


def parse_relational_query(description: str) -> tuple[str, str, str] | None:
    """Split "X <relation> Y" into (object, relation, reference), or None.

    Recognizes proximity ("next to", "near", "beside") and directional ("to the
    left/right of", "above", "below") relations, so the object instance can be
    disambiguated by its spatial relation to a reference object — relations YOLOE
    alone can't express. ``relation`` is one of ``near``/``left``/``right``/
    ``above``/``below``.
    """
    normalized = " ".join(description.strip().split()).strip(" ,.!?;:")
    match = _RELATION_RE.match(normalized)
    if not match:
        return None
    return (
        match.group(1).strip(" ,.!?;:"),
        _RELATIONS[match.group(2).lower()],
        match.group(3).strip(" ,.!?;:"),
    )


def select_by_nearest_to_reference(candidates: list[BBox], reference: BBox) -> BBox | None:
    """Return the candidate whose center is closest to the reference box's center."""
    return select_nearest_to_any_reference(candidates, [reference])


def select_nearest_to_any_reference(candidates: list[BBox], references: list[BBox]) -> BBox | None:
    """Return the candidate whose center is closest to ANY reference's center.

    Handles multiple reference instances ("the cup near the laptop" when there are
    two laptops): the object nearest to *whichever* reference is the natural match,
    not just the one nearest the highest-confidence reference.
    """
    if not candidates or not references:
        return None
    centers = [((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0) for r in references]

    def min_dist_sq(b: BBox) -> float:
        bx = (b[0] + b[2]) / 2.0
        by = (b[1] + b[3]) / 2.0
        return min((bx - cx) ** 2 + (by - cy) ** 2 for cx, cy in centers)

    return min(candidates, key=min_dist_sq)


def select_by_relation(candidates: list[BBox], relation: str, reference: BBox) -> BBox | None:
    """Pick the candidate satisfying a spatial `relation` to the `reference` box.

    ``near`` -> closest center. Directional relations (``left``/``right``/
    ``above``/``below``) keep only candidates on the correct side of the
    reference's center and return the one immediately adjacent to it; ``None`` if
    nothing is on that side (so the caller can fall back to the VLM).
    """
    if not candidates:
        return None
    if relation == "near":
        return select_by_nearest_to_reference(candidates, reference)

    rx = (reference[0] + reference[2]) / 2.0
    ry = (reference[1] + reference[3]) / 2.0

    def cx(b: BBox) -> float:
        return (b[0] + b[2]) / 2.0

    def cy(b: BBox) -> float:
        return (b[1] + b[3]) / 2.0

    if relation == "left":
        pool = [b for b in candidates if cx(b) < rx]
        return max(pool, key=cx) if pool else None  # immediately left of the ref
    if relation == "right":
        pool = [b for b in candidates if cx(b) > rx]
        return min(pool, key=cx) if pool else None
    if relation == "above":
        pool = [b for b in candidates if cy(b) < ry]
        return max(pool, key=cy) if pool else None
    if relation == "below":
        pool = [b for b in candidates if cy(b) > ry]
        return min(pool, key=cy) if pool else None
    return None


# Lower bound for the recall retry: still a real detection, not noise.
_RECALL_RETRY_CONFIDENCE = 0.25


def retry_at_lower_confidence(detector, image, description: str) -> list[BBox]:
    """Re-ground `description` once at a lower confidence; ``[]`` if not worthwhile.

    Uses ``process_image``'s per-call confidence override (no shared-state
    mutation, so it's safe under concurrency). Returns ``[]`` when the detector
    has no ``confidence`` knob or is already at/below the retry floor.
    """
    original = getattr(detector, "confidence", None)
    if original is None or original <= _RECALL_RETRY_CONFIDENCE:
        return []
    return ground_candidates_with_yoloe(
        detector, image, description, confidence=_RECALL_RETRY_CONFIDENCE
    )


def ground_candidates_for_completeness(detector, image, description: str) -> list[BBox]:
    """Ground all instances at a recall-favoring confidence (for "all the X").

    Multi-object grounding explicitly wants completeness, so detect at the lower
    recall threshold (measured on COCO128: many more real, well-localized
    instances at negligible precision cost) rather than the precision default.
    Uses the per-call confidence override (no shared-state mutation).
    """
    original = getattr(detector, "confidence", None)
    if original is None or original <= _RECALL_RETRY_CONFIDENCE:
        return ground_candidates_with_yoloe(detector, image, description)
    return ground_candidates_with_yoloe(
        detector, image, description, confidence=_RECALL_RETRY_CONFIDENCE
    )


def resolve_grounding(
    detector, image, description: str, *, prev_box: BBox | None = None
) -> BBox | None:
    """Ground `description`, disambiguating via the best available strategy.

    This is the unified entry point that composes the grounding primitives so a
    single natural-language query gets the right treatment:

    1. **Tracking** — if ``prev_box`` is given, return the candidate most
       consistent with it (:func:`select_nearest`), keeping a tracked target
       locked across frames.
    2. **Spatial** — if the query names a position ("the leftmost chair"),
       resolve it geometrically (:func:`select_by_position`).
    3. **Appearance** — for a multi-word phrase ("the red mug"), re-rank the
       candidates by CLIP similarity to the phrase (:func:`select_by_clip`),
       degrading to the top-confidence box if CLIP is unavailable.
    4. **Plain** — otherwise return the highest-confidence box.

    For an attributive phrase that the detector can't find directly, it retries
    with the head noun (last word) to improve open-vocab recall.

    Args:
        detector: A constructed open-vocab detector.
        image: A ``dimos.msgs.sensor_msgs.Image`` to ground against.
        description: The natural-language grounding query.
        prev_box: The target's previous-frame box, for tracking continuity.

    Returns:
        The chosen ``(x1, y1, x2, y2)`` bbox, or ``None`` if nothing matched.
    """
    # Relational queries ("the cup next to the laptop"): ground both the object
    # and the reference, then return the object instance closest to the reference.
    # Tracking (prev_box) takes priority and skips this.
    if prev_box is None:
        relational = parse_relational_query(description)
        if relational is not None:
            object_phrase, relation, reference_phrase = relational
            ref_class = _singularize_head(parse_grounding_query(reference_phrase)[0])
            obj_class = _singularize_head(parse_grounding_query(object_phrase)[0])

            # Use the same recall retry as the other paths so a faint reference or
            # object still resolves on the fast path instead of going to the VLM.
            obj_candidates = ground_candidates_with_yoloe(detector, image, obj_class)
            if not obj_candidates:
                obj_candidates = retry_at_lower_confidence(detector, image, obj_class)

            ref_candidates = ground_candidates_with_yoloe(detector, image, ref_class)
            if not ref_candidates:
                ref_candidates = retry_at_lower_confidence(detector, image, ref_class)

            if ref_candidates and obj_candidates:
                if relation == "near":
                    # Closest object to ANY reference instance (two laptops, etc.).
                    chosen = select_nearest_to_any_reference(obj_candidates, ref_candidates)
                else:
                    # Directional relations are ambiguous across multiple
                    # references; resolve against the highest-confidence one.
                    chosen = select_by_relation(obj_candidates, relation, ref_candidates[0])
                if chosen is not None:
                    return chosen
            # Reference/object missing or nothing on that side: VLM resolves it.
            return None

    object_phrase, qualifier = parse_grounding_query(description)

    # Singularize the head noun so plural queries ("the leftmost chairs") ground
    # the class YOLOE knows ("chair"); non-plural -s words are left intact.
    object_phrase = _singularize_head(object_phrase)

    candidates = ground_candidates_with_yoloe(detector, image, object_phrase)
    if not candidates and _is_attributive(object_phrase):
        # Open-vocab recall fallback: try the bare, singularized head noun, e.g.
        # "mug" for "red mug" / "red mugs", when the full phrase found nothing.
        head_noun = singularize(object_phrase.split()[-1])
        candidates = ground_candidates_with_yoloe(detector, image, head_noun)
    if not candidates:
        # Last cheap try before the caller falls to the slow VLM: re-run YOLOE at
        # a lower confidence to catch a faint/small object it skipped. Only fires
        # on a miss, so the precision-oriented default is unchanged for hits.
        candidates = retry_at_lower_confidence(detector, image, object_phrase)
    if not candidates:
        return None

    if prev_box is not None:
        return select_nearest(candidates, prev_box)
    if qualifier is not None:
        return select_by_position(candidates, qualifier)
    if len(candidates) > 1 and _is_attributive(object_phrase):
        try:
            return select_by_clip(image, candidates, description)
        except Exception:
            logger.warning(
                "CLIP attribute re-ranking unavailable; using top-confidence match.",
                exc_info=True,
            )
            return candidates[0]
    return candidates[0]
