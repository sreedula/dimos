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

"""Measure SPATIAL grounding accuracy on real COCO128 annotations.

The unit tests cover ``select_by_position`` with synthetic boxes; this validates
the end-to-end spatial path ("the leftmost person" -> YOLOE detection -> leftmost
selection) against COCO's human box annotations.

For every (image, class) with >= 2 ground-truth instances, it grounds
"the leftmost <class>" and "the rightmost <class>" via the real ``resolve_grounding``
pipeline and checks the returned box best-matches (highest IoU) the GT instance
that is actually leftmost / rightmost by center-x, at IoU >= 0.5. This is a
combined detection+selection metric: if YOLOE misses the extreme instance, the
query can't return it, and that counts (honestly) as a miss.

Reuses the COCO128 plumbing from ``eval_grounding`` so there is one source of
truth for download/labels. CPU-only; reproduce with ``python eval_spatial.py``.
"""

from __future__ import annotations

import os

from dimos.navigation.visual.grounding import build_yoloe_grounding_detector, resolve_grounding
from eval_grounding import (
    COCO128_DIR,
    EVAL_CLASS_NAMES,
    MAX_IMAGES,
    coco_names,
    ensure_coco128,
    iou,
    load_bgr_image,
    load_gt_boxes,
)


def _center_x(box) -> float:
    return (box[0] + box[2]) / 2.0


def _matches_extreme(grounded, gt_boxes, extreme_idx: int) -> bool:
    """True if `grounded` best-matches the `extreme_idx` GT box (IoU >= 0.5)."""
    if grounded is None:
        return False
    ious = [iou(grounded, gt) for gt in gt_boxes]
    best = max(range(len(gt_boxes)), key=lambda i: ious[i])
    return best == extreme_idx and ious[best] >= 0.5


def main() -> int:
    confidence = float(os.environ.get("EVAL_CONFIDENCE", "0.6"))
    print(f"Building YOLOE grounding detector (CPU, confidence={confidence})...")
    detector = build_yoloe_grounding_detector(confidence=confidence)
    if detector is None:
        print("FATAL: YOLOE detector unavailable; cannot evaluate.")
        return 1

    if not ensure_coco128():
        print("FATAL: COCO128 unavailable; cannot run the spatial eval.")
        return 1

    names = coco_names()
    name_to_id = {n: i for i, n in enumerate(names)}
    wanted_ids = {name_to_id[n] for n in EVAL_CLASS_NAMES if n in name_to_id}

    img_dir = COCO128_DIR / "images" / "train2017"
    lbl_dir = COCO128_DIR / "labels" / "train2017"
    image_paths = sorted(img_dir.glob("*.jpg"))[:MAX_IMAGES]
    print(f"Evaluating spatial grounding over {len(image_paths)} COCO128 images.\n")

    n = left_ok = right_ok = 0
    for path in image_paths:
        try:
            img = load_bgr_image(path)
        except Exception:
            continue
        gt_by_class = load_gt_boxes(
            lbl_dir / (path.stem + ".txt"), img.width, img.height, wanted_ids
        )
        for cls, gt_boxes in gt_by_class.items():
            if len(gt_boxes) < 2:
                continue  # spatial selection only meaningful with >= 2 instances
            name = names[cls]
            order = sorted(range(len(gt_boxes)), key=lambda i: _center_x(gt_boxes[i]))
            leftmost_idx, rightmost_idx = order[0], order[-1]
            n += 1
            left = resolve_grounding(detector, img, f"the leftmost {name}")
            right = resolve_grounding(detector, img, f"the rightmost {name}")
            left_ok += _matches_extreme(left, gt_boxes, leftmost_idx)
            right_ok += _matches_extreme(right, gt_boxes, rightmost_idx)

    print("=" * 70)
    print(f"Multi-instance (image, class) pairs evaluated: {n}")
    if n:
        print(f"  leftmost  correct: {left_ok}/{n} ({left_ok / n:.0%})")
        print(f"  rightmost correct: {right_ok}/{n} ({right_ok / n:.0%})")
        print(
            f"  combined:          {(left_ok + right_ok)}/{2 * n} ({(left_ok + right_ok) / (2 * n):.0%})"
        )
    print("=" * 70)
    print(
        "Combined detection+selection: a miss can mean YOLOE didn't detect the\n"
        "extreme instance (not a selection error). CPU, conf=0.6 default."
    )
    detector.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
