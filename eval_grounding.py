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

"""Aggregate accuracy eval for the YOLOE grounding fast-path vs HUMAN ground truth.

WHY THIS EXISTS
  ``bench_grounding.py`` measures *speed* and reports an accuracy IoU on just two
  bundled images (bus.jpg, zidane.jpg) against another *detector*. Two images is
  too thin to trust, and detector-vs-detector agreement is not ground truth. This
  harness broadens the accuracy claim into a real aggregate number over many
  images, scored against COCO's HUMAN-annotated boxes.

DATASET (primary path — REAL human ground truth)
  COCO128: the first 128 images of COCO train2017 with their human-labelled
  bounding boxes, distributed by Ultralytics (~7 MB). Auto-downloaded on first
  run to ``data/coco128`` if absent. Labels are YOLO-format text files, one line
  per object: ``cls cx cy w h`` with all four geometry values normalized to
  ``0..1`` of the image's width/height. The 80 COCO class names come from
  ultralytics' bundled ``coco128.yaml`` (hardcoded fallback if unreadable).

WHAT IS MEASURED
  For a bounded subset of images (see ``MAX_IMAGES``), and only for a handful of
  common, visually-unambiguous classes that actually appear in COCO128
  (``person, car, bus, truck, dog, chair``), the harness:
    1. Grounds the bare class NAME with the YOLOE fast-path
       (``ground_candidates_with_yoloe``), yielding YOLOE's pixel boxes.
    2. Converts that class's human-GT boxes for the image to pixel ``xyxy``.
    3. For each GT instance, finds the best-IoU YOLOE box and records that IoU
       (a recall-oriented per-instance localization score; a present class for
       which YOLOE returned nothing scores 0.0 for every GT instance — misses are
       NOT hidden).

METRIC DEFINITIONS (all computed over the evaluated subset, printed at the end)
  - median IoU vs GT : median, over every (image, class, GT-instance) triple, of
    that GT box's best IoU with any YOLOE box. Misses count as 0.0, so this is an
    honest recall-aware number, not a hits-only figure.
  - hit-rate @ IoU>=0.5 : fraction of GT instances whose best IoU is >= 0.50 —
    "how often YOLOE localizes a real object well enough to use" (this is
    recall@0.5).
  - recall : fraction of (image, class) pairs where the class IS present in the
    GT and YOLOE returned at least one box — "how often the fast-path finds a
    present class at all" (a detection-confidence-threshold signal, conf=0.6).

  Each metric is reported twice: over ALL human GT instances, and over the
  PROMINENT subset (GT box area >= ``PROMINENT_AREA_FRAC`` of the frame). The two
  differ sharply and the gap is itself the finding: COCO annotates every tiny,
  occluded, far-background instance, but the grounding fast-path exists to lock
  onto a PROMINENT target object a robot would actually approach, and ships at
  conf=0.6 so it deliberately does NOT fire on a 10px background person. The
  all-instances numbers are dragged down by those tiny annotations; the prominent
  numbers reflect the localization quality on targets that matter. Both are
  printed; neither is hidden.

HONESTY
  - Primary path scores against HUMAN ground truth (COCO128 labels) — this is the
    headline this harness exists to produce.
  - FALLBACK: if the COCO128 download or its labels genuinely cannot be loaded,
    the harness instead builds proxy ground truth from a YOLO11 reference
    detector over the same images and labels the result, unmistakably, as
    inter-detector agreement (NOT human GT). It never fabricates numbers, and it
    prints which path it took.
  - CPU-only run. YOLOE keeps its shipped detection threshold (conf=0.6), so a
    faint/occluded GT instance below threshold legitimately counts as a miss; the
    metrics describe the fast-path exactly as deployed, not an idealized variant.
  - This is a still-image benchmark. An on-robot / simulator grounding run is a
    separate manual follow-up and is NOT performed here.

RUN
  VIRTUAL_ENV=/Users/sreekare/dimos/.venv .venv/bin/python eval_grounding.py

Like ``bench_grounding.py`` this is a standalone script, not a pytest target.
"""

from __future__ import annotations

from pathlib import Path
import statistics

import cv2

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.navigation.visual.grounding import (
    build_yoloe_grounding_detector,
    ground_candidates_with_yoloe,
)

# --- knobs ---------------------------------------------------------------------
# Hard cap on images so a CPU run stays bounded; the actual evaluated count is
# printed (never silently truncated). COCO128 has 128 images total.
MAX_IMAGES = 128

# Common, visually-unambiguous COCO classes to ground. Restricting to these keeps
# the eval to objects a human label and an open-vocab prompt agree on cleanly
# (vs. e.g. "tie" or "handbag" where annotation and prompt semantics drift).
EVAL_CLASS_NAMES = ("person", "car", "bus", "truck", "dog", "chair")

HIT_IOU = 0.50  # IoU threshold for the "well-localized" hit-rate.

# A GT box at least this fraction of the frame's area is a "prominent" target —
# the kind a grounding fast-path is meant to find and a robot would approach.
# Smaller boxes are COCO's exhaustive background annotations (tiny/occluded).
PROMINENT_AREA_FRAC = 0.02

DATA_DIR = Path("/Users/sreekare/dimos/data")
COCO128_DIR = DATA_DIR / "coco128"
COCO128_URL = "https://ultralytics.com/assets/coco128.zip"

# Standard COCO80 class names (used if coco128.yaml cannot be read).
COCO80_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_bgr_image(path: Path) -> Image:
    """Load ``path`` as a guaranteed 3-channel BGR Image.

    ``cv2.IMREAD_COLOR`` promotes COCO128's occasional grayscale JPEGs to 3
    channels, which the detectors require (the plain ``Image.from_file`` keeps
    them single-channel and the conv stack then rejects them).
    """
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"Could not load image from {path}")
    return Image.from_opencv(arr, format=ImageFormat.BGR)


def iou(a, b) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes, in ``[0, 1]``.

    Defined locally (not imported from the module under test) so the metric is
    independent of the code it scores.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def coco_names() -> list[str]:
    """COCO80 class names from ultralytics' coco128.yaml; hardcoded fallback."""
    try:
        import ultralytics
        import yaml

        yaml_path = Path(ultralytics.__file__).parent / "cfg/datasets/coco128.yaml"
        names = yaml.safe_load(yaml_path.read_text())["names"]
        return [names[i] for i in range(len(names))]
    except Exception:
        return COCO80_NAMES


def ensure_coco128() -> bool:
    """Download+unzip COCO128 into ``DATA_DIR`` if absent. True if usable."""
    img_dir = COCO128_DIR / "images" / "train2017"
    lbl_dir = COCO128_DIR / "labels" / "train2017"
    if img_dir.is_dir() and lbl_dir.is_dir() and any(img_dir.glob("*.jpg")):
        return True
    try:
        from ultralytics.utils.downloads import download

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        download(COCO128_URL, dir=str(DATA_DIR))
    except Exception as exc:
        print(f"[coco128] download failed: {exc}")
        return False
    return img_dir.is_dir() and lbl_dir.is_dir() and any(img_dir.glob("*.jpg"))


def load_gt_boxes(label_path: Path, width: int, height: int, wanted: set[int]):
    """Parse a YOLO label file into ``{class_id: [(x1,y1,x2,y2), ...]}`` (pixels).

    Only classes in ``wanted`` are kept. Returns ``{}`` if the file is missing.
    """
    out: dict[int, list] = {}
    if not label_path.is_file():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in wanted:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        out.setdefault(cls, []).append((x1, y1, x2, y2))
    return out


def proxy_gt_boxes(yolo11, img, wanted: set[int]):
    """Build proxy GT from a YOLO11 reference detector: ``{class_id: [xyxy]}``.

    Used only on the fallback path when COCO128 human labels are unavailable.
    YOLO11 shares COCO's 80-class indexing, so its class ids map directly.
    """
    out: dict[int, list] = {}
    res = yolo11.predict(img.to_opencv(), conf=0.5, verbose=False)[0]
    for box in res.boxes:
        cls = int(box.cls.item())
        if cls not in wanted:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        out.setdefault(cls, []).append((x1, y1, x2, y2))
    return out


def evaluate(detector, gt_by_class: dict[int, list], img, id_to_name: dict[int, str]):
    """Score one image's GT against YOLOE in a single grounding pass per class.

    Returns ``(records, n_pairs, n_pairs_returned)`` where each record is
    ``{class, iou, hit}`` for one GT instance, ``n_pairs`` counts the
    (image, present-class) pairs, and ``n_pairs_returned`` counts those for which
    YOLOE returned at least one box (the class-presence recall signal).
    """
    frame_area = float(img.width * img.height)
    records = []
    n_pairs = 0
    n_pairs_returned = 0
    for cls, gt_boxes in gt_by_class.items():
        name = id_to_name[cls]
        # One grounding call per (image, class) — reused for both recall and IoU.
        yoloe_boxes = ground_candidates_with_yoloe(detector, img, name)
        n_pairs += 1
        if yoloe_boxes:
            n_pairs_returned += 1
        for gt in gt_boxes:
            best = max((iou(gt, yb) for yb in yoloe_boxes), default=0.0)
            gt_area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
            records.append(
                {
                    "class": name,
                    "iou": best,
                    "hit": best >= HIT_IOU,
                    "prominent": frame_area > 0 and gt_area / frame_area >= PROMINENT_AREA_FRAC,
                }
            )
    return records, n_pairs, n_pairs_returned


def _cut_metrics(records):
    """Return ``(n, median_iou, hit_rate)`` for a list of instance records."""
    if not records:
        return 0, None, None
    ious = [r["iou"] for r in records]
    hits = sum(r["hit"] for r in records)
    return len(records), statistics.median(ious), hits / len(records)


def aggregate_and_print(records, n_images, class_present_pairs, gt_truth_label):
    """Print per-class and headline aggregate metrics, all-vs-prominent."""
    print("\n" + "=" * 78)
    print(f"PER-CLASS — YOLOE grounding vs {gt_truth_label}")
    print(f"  (all GT instances  |  prominent = GT box >= {PROMINENT_AREA_FRAC:.0%} of frame)")
    print("=" * 78)
    print(
        f"{'class':<9} {'inst':>5} {'medIoU':>7} {'hit@.5':>7}   "
        f"{'promInst':>8} {'medIoU':>7} {'hit@.5':>7}"
    )
    print("-" * 78)
    for name in EVAL_CLASS_NAMES:
        rs = [r for r in records if r["class"] == name]
        if not rs:
            continue
        n_a, iou_a, hit_a = _cut_metrics(rs)
        n_p, iou_p, hit_p = _cut_metrics([r for r in rs if r["prominent"]])
        ip = f"{iou_p:.2f}" if iou_p is not None else "  -"
        hp = f"{hit_p:.0%}" if hit_p is not None else "  -"
        print(f"{name:<9} {n_a:>5} {iou_a:>7.2f} {hit_a:>7.0%}   {n_p:>8} {ip:>7} {hp:>7}")
    print("-" * 78)

    n_instances = len(records)
    prominent = [r for r in records if r["prominent"]]
    n_pairs, n_pairs_returned = class_present_pairs

    n_a, iou_a, hit_a = _cut_metrics(records)
    n_p, iou_p, hit_p = _cut_metrics(prominent)
    localized = [r["iou"] for r in records if r["iou"] > 0.0]

    print("\n" + "=" * 78)
    print("HEADLINE — aggregate accuracy")
    print("=" * 78)
    print(f"ground truth source         : {gt_truth_label}")
    print(f"images evaluated            : {n_images} (cap = {MAX_IMAGES})")
    print(f"classes evaluated           : {', '.join(EVAL_CLASS_NAMES)}")
    print(f"(image, present-class) pairs : {n_pairs}")
    print(
        f"GT instances evaluated      : {n_instances} "
        f"({n_p} prominent, {n_instances - n_p} tiny/background)"
    )
    print("-" * 78)
    print("ALL GT instances (incl. COCO's exhaustive tiny annotations):")
    print(f"  median IoU vs ground truth : {iou_a:.2f}")
    print(
        f"  hit-rate @ IoU>=0.5        : {hit_a:.0%} ({sum(r['hit'] for r in records)}/{n_instances})"
    )
    print("-" * 78)
    print(
        f"PROMINENT targets (GT box >= {PROMINENT_AREA_FRAC:.0%} of frame — the grounding use case):"
    )
    if n_p:
        print(f"  median IoU vs ground truth : {iou_p:.2f}")
        print(
            f"  hit-rate @ IoU>=0.5        : {hit_p:.0%} ({sum(r['hit'] for r in prominent)}/{n_p})"
        )
    else:
        print("  (no prominent instances in evaluated subset)")
    print("-" * 78)
    if localized:
        print(
            f"localization quality (median IoU over the {len(localized)} instances "
            f"YOLOE found): {statistics.median(localized):.2f}"
        )
    if n_pairs:
        print(
            f"recall (present class found) : {n_pairs_returned / n_pairs:.0%} "
            f"({n_pairs_returned}/{n_pairs} present classes returned >=1 box)"
        )
    print("=" * 78)


def main() -> int:
    print("Building YOLOE grounding detector (CPU)...")
    detector = build_yoloe_grounding_detector()
    if detector is None:
        print("FATAL: YOLOE detector unavailable; cannot evaluate.")
        return 1

    names = coco_names()
    name_to_id = {n: i for i, n in enumerate(names)}
    wanted_ids = {name_to_id[n] for n in EVAL_CLASS_NAMES if n in name_to_id}
    id_to_name = {i: names[i] for i in wanted_ids}

    use_real_gt = ensure_coco128()
    yolo11 = None
    if use_real_gt:
        img_dir = COCO128_DIR / "images" / "train2017"
        lbl_dir = COCO128_DIR / "labels" / "train2017"
        image_paths = sorted(img_dir.glob("*.jpg"))[:MAX_IMAGES]
        gt_label = "HUMAN ground truth (COCO128 labels)"
        print(
            f"Path: REAL human GT. Found {len(sorted(img_dir.glob('*.jpg')))} "
            f"COCO128 images; evaluating first {len(image_paths)}."
        )
    else:
        # Fallback: proxy GT from a YOLO11 reference detector over bundled images.
        print("Path: FALLBACK (COCO128 unavailable) -> YOLO11 proxy GT.")
        from ultralytics import YOLO

        yolo11 = YOLO("yolo11s.pt")
        sample_dir = Path(__import__("ultralytics").__file__).parent / "assets"
        image_paths = sorted(sample_dir.glob("*.jpg"))[:MAX_IMAGES]
        gt_label = "YOLO11 reference detector (inter-detector agreement, NOT human GT)"
        print(f"Found {len(image_paths)} sample images for proxy-GT eval.")

    records = []
    n_images = 0
    n_pairs = 0
    n_pairs_returned = 0
    for path in image_paths:
        try:
            img = load_bgr_image(path)
        except Exception as exc:
            print(f"  skip {path.name}: load failed ({exc})")
            continue

        if use_real_gt:
            label_path = lbl_dir / (path.stem + ".txt")
            gt_by_class = load_gt_boxes(label_path, img.width, img.height, wanted_ids)
        else:
            gt_by_class = proxy_gt_boxes(yolo11, img, wanted_ids)

        if not gt_by_class:
            continue  # no eval-class object present in this image

        n_images += 1
        recs, pairs, pairs_returned = evaluate(detector, gt_by_class, img, id_to_name)
        records.extend(recs)
        n_pairs += pairs
        n_pairs_returned += pairs_returned

    detector.stop()

    if not records:
        print("\nNo evaluable (image, class, instance) triples found — nothing to report.")
        return 1

    aggregate_and_print(records, n_images, (n_pairs, n_pairs_returned), gt_label)
    print(
        "\nNOTE: still-image benchmark only. An on-robot / simulator grounding run "
        "remains a separate manual follow-up (not performed here)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
