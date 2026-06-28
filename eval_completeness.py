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

"""Measure multi-object grounding PRECISION and recall on COCO128.

`eval_grounding.py` measures recall (fraction of GT instances recovered) and the
localization IoU of the boxes YOLOE returns. It does NOT cleanly measure
precision — whether the boxes returned for "all the X" actually ARE X. That
matters for the multi-object completeness path (`ground_all_instances`), which
detects at a recall-favoring confidence (0.25) and could in principle add false
positives. This script measures both, at 0.6 and 0.25, so the completeness
confidence choice can be judged on precision, not just recall.

For each (image, class) with >= 1 COCO GT instance:
  - recall:    fraction of GT boxes matched (IoU >= 0.5) by some grounded box.
  - precision: fraction of grounded boxes that match (IoU >= 0.5) some GT box.

COCO128 is exhaustively annotated, so a grounded box matching no GT is a genuine
false positive (or a tiny instance). CPU-only; `EVAL_CONFIDENCE` configurable.
"""

from __future__ import annotations

import os

from dimos.navigation.visual.grounding import build_yoloe_grounding_detector, ground_all_instances
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


def main() -> int:
    confidence = float(os.environ.get("EVAL_CONFIDENCE", "0.6"))
    print(f"Building YOLOE grounding detector (CPU, confidence={confidence})...")
    detector = build_yoloe_grounding_detector(confidence=confidence)
    if detector is None:
        print("FATAL: YOLOE detector unavailable; cannot evaluate.")
        return 1
    if not ensure_coco128():
        print("FATAL: COCO128 unavailable; cannot run the completeness eval.")
        return 1

    names = coco_names()
    name_to_id = {n: i for i, n in enumerate(names)}
    wanted_ids = {name_to_id[n] for n in EVAL_CLASS_NAMES if n in name_to_id}

    img_dir = COCO128_DIR / "images" / "train2017"
    lbl_dir = COCO128_DIR / "labels" / "train2017"
    image_paths = sorted(img_dir.glob("*.jpg"))[:MAX_IMAGES]
    print(f"Evaluating multi-object precision/recall over {len(image_paths)} images.\n")

    gt_total = gt_hit = grounded_total = grounded_hit = 0
    for path in image_paths:
        try:
            img = load_bgr_image(path)
        except Exception:
            continue
        gt_by_class = load_gt_boxes(
            lbl_dir / (path.stem + ".txt"), img.width, img.height, wanted_ids
        )
        for cls, gt_boxes in gt_by_class.items():
            grounded = ground_all_instances(detector, img, names[cls])
            # recall: each GT matched by some grounded box.
            for gt in gt_boxes:
                gt_total += 1
                gt_hit += any(iou(gt, gb) >= 0.5 for gb in grounded)
            # precision: each grounded box matches some GT.
            for gb in grounded:
                grounded_total += 1
                grounded_hit += any(iou(gb, gt) >= 0.5 for gt in gt_boxes)

    print("=" * 70)
    print(f"GT instances: {gt_total} | grounded boxes: {grounded_total}")
    if gt_total:
        print(f"  recall    (GT recovered)        : {gt_hit}/{gt_total} ({gt_hit / gt_total:.0%})")
    if grounded_total:
        print(
            f"  precision (grounded that are GT): "
            f"{grounded_hit}/{grounded_total} ({grounded_hit / grounded_total:.0%})"
        )
    print("=" * 70)
    print("Note: ground_all_instances grounds at min(confidence, 0.25), so any")
    print("EVAL_CONFIDENCE >= 0.25 measures the same 0.25 completeness floor.")
    print("COCO128 is exhaustively annotated, so a grounded box matching no GT is")
    print("a genuine false positive (or a tiny/occluded instance). CPU.")
    detector.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
