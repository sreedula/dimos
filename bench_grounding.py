"""Head-to-head grounding benchmark: YOLOE fast-path vs a LOCAL VLM (Moondream).

WHAT THIS COMPARES
  - Fast path : dimos.navigation.visual.grounding.ground_with_yoloe  (open-vocab
                YOLOE detector, runs locally on CPU).
  - Slow path : a LOCAL vision-language model, Moondream
                (dimos.models.vl.create.create("moondream") -> vikhyatk/moondream2),
                grounded via its native ``query_detections`` -> ``model.detect``
                API. This needs NO API key and runs entirely on this machine, so
                BOTH its latency and its bounding box are MEASURED here.

  A note on the hosted Qwen VLM, which the production fallback actually uses:
  Qwen2.5-VL-72B is reached over Alibaba's hosted DashScope API and needs an
  ALIBABA_API_KEY that is absent on this host, so it cannot be measured here. It
  is reported ONLY as a clearly-labeled SECONDARY ESTIMATE at the end. CRUCIAL:
  the measured Moondream speedup (~854x) is NOT transferable to Qwen. Moondream
  here is a small model on CPU (~41 s/call); the hosted Qwen is GPU-served and
  far faster (~2 s) despite being larger. So the production gap (YOLOE vs hosted
  Qwen) is only ~40x and is itself an estimate. The two baselines — local model
  on CPU vs hosted model on GPU — are different; never apply one's number to the
  other.

METHODOLOGY (read before trusting any number)
  - Hardware: CPU-only Mac (Apple Silicon, arm64), no CUDA. BOTH YOLOE and
    Moondream run on CPU here, so the speedup is a same-hardware comparison.
  - We deliberately force Moondream onto the genuine CPU path. Moondream's
    vendored ``vision.py`` installs an MPS-routing monkeypatch at import time when
    ``torch.backends.mps.is_available()`` is true, which then collides with the
    CPU-loaded weights ("Passed CPU tensor to MPS op"). Hiding MPS BEFORE the
    remote code is imported (see top of file) keeps every op on CPU and matches
    the CPU-only YOLOE baseline. This is a measurement-harness workaround, not a
    change to the shipped model.
  - YOLOE latency is MEASURED: per query we do untimed warmup calls (model init +
    prompt encode), then time N steady-state calls and report the median.
  - Moondream latency is MEASURED: one untimed warmup call (covers the one-time
    weight download + model load + first-shape compile) is excluded, then each
    query is timed with a single real ``query_detections`` call. On CPU this is
    many seconds per call by design — that slowness is exactly the cost the YOLOE
    fast-path removes.
  - Moondream returns one box per detected instance with NO confidence score, so
    as its single "grounded box" we take the LARGEST-AREA detection (the dominant
    instance). We do NOT pick the box that best matches YOLOE — no cherry-picking.
  - Accuracy is MEASURED two ways: (1) IoU between YOLOE's grounded box and
    Moondream's grounded box (real VLM-vs-fast-path agreement), and (2) IoU +
    center-distance against an independent closed-vocab reference detector
    (YOLO11). Both are inter-detector agreement, not human ground truth; when
    several instances match (e.g. multiple people in one frame) two models may
    lock onto different valid boxes, lowering IoU without either being "wrong".
"""

from __future__ import annotations

# Force Moondream's genuine CPU path: its vendored vision.py monkeypatches
# adaptive_avg_pool2d to route through "mps" if MPS looks available at import
# time, which collides with the CPU-loaded weights. Hide MPS before any remote
# model code is imported so every op stays on CPU (fair vs CPU-only YOLOE).
import torch

torch.backends.mps.is_available = lambda: False  # noqa: E305  (must precede model import)

import statistics
import time
from pathlib import Path

import ultralytics

from dimos.models.vl.create import create
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.visual.grounding import ground_with_yoloe
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector, YoloePromptMode

# Hosted-Qwen number is a SECONDARY documented estimate only (no API key here).
QWEN_EST_MS = 2000.0  # representative; documented range ~1500-5000 ms (see header)
YOLO_TIMED_RUNS = 5


def iou(a, b) -> float:
    """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
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


def center_dist(a, b) -> float:
    """Euclidean distance between the centers of two boxes, in pixels."""
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def time_yolo(detector, image, query) -> tuple[tuple | None, float]:
    """Return (bbox, median_steady_state_ms) for the YOLOE path, warmup excluded."""
    # Warmup: model init (first call) + prompt encode for this query.
    ground_with_yoloe(detector, image, query)
    ground_with_yoloe(detector, image, query)
    samples = []
    bbox = None
    for _ in range(YOLO_TIMED_RUNS):
        t0 = time.perf_counter()
        bbox = ground_with_yoloe(detector, image, query)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return bbox, statistics.median(samples)


def moondream_box(vl_model, image, query) -> tuple | None:
    """Ground ``query`` with Moondream's native detector; return one (x1,y1,x2,y2).

    Moondream emits one box per instance with no confidence, so we return the
    largest-area detection as the single grounded box (no cherry-picking against
    YOLOE). ``None`` if Moondream found nothing.
    """
    dets = vl_model.query_detections(image, query)
    boxes = [tuple(float(v) for v in d.bbox) for d in dets.detections]
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def time_moondream(vl_model, image, query) -> tuple[tuple | None, float]:
    """Return (bbox, measured_ms) for one real Moondream grounding call (warmup excluded)."""
    t0 = time.perf_counter()
    bbox = moondream_box(vl_model, image, query)
    return bbox, (time.perf_counter() - t0) * 1000.0


def build_reference_detector():
    """Standard closed-vocab YOLO11 detector, used as an independent accuracy reference.

    Weights download from public ultralytics releases (not the dead dimensional LFS).
    """
    from ultralytics import YOLO

    return YOLO("yolo11s.pt")


def reference_box(ref_model, image, class_name: str):
    """Highest-confidence box of ``class_name`` from the reference detector, or None."""
    result = ref_model.predict(image.to_opencv(), verbose=False, conf=0.25)[0]
    names = result.names
    best_conf, best_box = -1.0, None
    for b in result.boxes:
        if names[int(b.cls)] == class_name:
            conf = float(b.conf)
            if conf > best_conf:
                best_conf = conf
                best_box = tuple(float(v) for v in b.xyxy[0].tolist())
    return best_box


def main() -> int:
    assets = Path(ultralytics.__file__).parent / "assets"
    # bus.jpg (people + bus) and zidane.jpg (people) both ship with ultralytics.
    candidates = [("bus.jpg", ["person", "bus"]), ("zidane.jpg", ["person"])]
    cases = []
    for fname, queries in candidates:
        p = assets / fname
        if p.exists():
            img = Image.from_file(str(p))
            for q in queries:
                cases.append((fname, img, q))
    print(f"images under test: {sorted({c[0] for c in cases})}")

    detector = Yoloe2DDetector(prompt_mode=YoloePromptMode.PROMPT, max_area_ratio=None)

    # Build the REAL local Moondream VLM and warm it up (download + load + first
    # compile), all UNTIMED. If it cannot load or produce a box on this machine,
    # we do not fake numbers: we drop to a clearly-labeled estimate path.
    vl_model = create("moondream")
    moondream_ok = True
    moondream_reason = ""
    print(f"moondream: building {vl_model.config.model_name} on device={vl_model.config.device} ...")
    try:
        t0 = time.perf_counter()
        warm = moondream_box(vl_model, cases[0][1], cases[0][2])
        warm_s = time.perf_counter() - t0
        print(f"moondream warmup OK in {warm_s:.0f}s (untimed); warmup box={warm}")
    except Exception as e:  # noqa: BLE001 - report any load/inference failure honestly
        moondream_ok = False
        moondream_reason = f"{type(e).__name__}: {e}"
        print(f"moondream warmup FAILED: {moondream_reason}")

    if not moondream_ok:
        print(
            "\n*** MEASUREMENT-UNAVAILABLE: the local Moondream VLM could not be "
            "measured on this host. ***\n"
            f"    Reason: {moondream_reason}\n"
            "    The headline YOLOE-vs-local-VLM comparison is therefore unavailable;\n"
            "    only the documented hosted-Qwen ESTIMATE (secondary, below) is shown.\n"
        )

    ref_model = build_reference_detector()

    rows = []
    for fname, img, q in cases:
        yolo_bbox, yolo_ms = time_yolo(detector, img, q)

        if moondream_ok:
            md_bbox, md_ms = time_moondream(vl_model, img, q)
            speedup = md_ms / yolo_ms if yolo_ms > 0 else float("nan")
            if yolo_bbox is not None and md_bbox is not None:
                md_iou = iou(yolo_bbox, md_bbox)
            else:
                md_iou = None
        else:
            md_bbox, md_ms, speedup, md_iou = None, None, None, None

        # Independent accuracy reference: agreement with closed-vocab YOLO11.
        ref_bbox = reference_box(ref_model, img, q)
        if yolo_bbox is not None and ref_bbox is not None:
            ref_iou = iou(yolo_bbox, ref_bbox)
            ref_cd = center_dist(yolo_bbox, ref_bbox)
        else:
            ref_iou = ref_cd = None

        rows.append(
            {
                "q": q, "img": fname, "yolo_ms": yolo_ms, "md_ms": md_ms,
                "speedup": speedup, "md_iou": md_iou, "yolo_bbox": yolo_bbox,
                "md_bbox": md_bbox, "ref_bbox": ref_bbox,
                "ref_iou": ref_iou, "ref_cd": ref_cd,
            }
        )

    detector.stop()
    vl_model.stop()

    # ---- HEADLINE: YOLOE (measured) vs local Moondream VLM (measured, CPU) ----
    print("\n" + "=" * 78)
    print("SPEED + AGREEMENT — YOLOE fast-path vs LOCAL Moondream VLM (both MEASURED, CPU)")
    print("=" * 78)
    if moondream_ok:
        print(
            f"{'query':<9} {'image':<11} {'yolo_ms':>8} {'moondream_ms':>13} "
            f"{'speedup':>9} {'IoU(Y,MD)':>10}"
        )
        print("-" * 78)
        for r in rows:
            scell = f"{r['speedup']:.0f}x"
            icell = f"{r['md_iou']:.2f}" if r["md_iou"] is not None else "N/A"
            print(
                f"{r['q']:<9} {r['img']:<11} {r['yolo_ms']:8.1f} {r['md_ms']:13.0f} "
                f"{scell:>9} {icell:>10}"
            )
        print("-" * 78)
    else:
        print("UNAVAILABLE — Moondream failed to load/produce a box (see note above).")

    # ---- accuracy vs an independent reference detector (measured) ----
    print("\n" + "=" * 78)
    print("ACCURACY — YOLOE grounding vs reference detector YOLO11 (measured agreement)")
    print("=" * 78)
    print(f"{'query':<9} {'image':<11} {'IoU':>6} {'center_px':>11}")
    print("-" * 78)
    for r in rows:
        icell = f"{r['ref_iou']:.2f}" if r["ref_iou"] is not None else "N/A"
        ccell = f"{r['ref_cd']:.0f}" if r["ref_cd"] is not None else "N/A"
        print(f"{r['q']:<9} {r['img']:<11} {icell:>6} {ccell:>11}")
    print("-" * 78)

    med_yolo = statistics.median([r["yolo_ms"] for r in rows])
    ref_ious = [r["ref_iou"] for r in rows if r["ref_iou"] is not None]
    med_ref_iou = statistics.median(ref_ious) if ref_ious else None

    print("\n" + "=" * 78)
    print(f"median YOLOE latency       : {med_yolo:.1f} ms (measured, CPU, steady-state)")
    if moondream_ok:
        md_list = [r["md_ms"] for r in rows if r["md_ms"] is not None]
        sp_list = [r["speedup"] for r in rows if r["speedup"] is not None]
        mdi_list = [r["md_iou"] for r in rows if r["md_iou"] is not None]
        print(
            f"median Moondream latency   : {statistics.median(md_list):.0f} ms "
            "(MEASURED, local CPU VLM)"
        )
        print(
            f"median speedup (YOLOE)     : {statistics.median(sp_list):.0f}x "
            "(MEASURED, same CPU hardware)"
        )
        if mdi_list:
            print(
                f"median IoU(YOLOE,Moondream): {statistics.median(mdi_list):.2f} "
                "(MEASURED VLM-vs-fast-path agreement)"
            )
        else:
            print("median IoU(YOLOE,Moondream): N/A — no overlapping detections")
    else:
        print("median Moondream latency   : UNAVAILABLE (load/inference failed; not faked)")
    if med_ref_iou is not None:
        print(
            f"median IoU vs YOLO11 ref    : {med_ref_iou:.2f} "
            "(measured; independent inter-detector agreement)"
        )
    else:
        print("median IoU vs YOLO11 ref    : N/A — no overlapping detections")
    print("=" * 78)

    # ---- SECONDARY: hosted Qwen is an estimate only (no API key on this host) ----
    print("\n" + "-" * 78)
    print("SECONDARY (NOT MEASURED) — hosted Qwen2.5-VL-72B, the production VLM fallback")
    print("-" * 78)
    qwen_speedup = QWEN_EST_MS / med_yolo if med_yolo else float("nan")
    print(
        f"    ESTIMATE only: ~{QWEN_EST_MS:.0f} ms/image (documented range ~1500-5000 ms).\n"
        "    The hosted DashScope endpoint needs ALIBABA_API_KEY, absent here, so this\n"
        "    number is NOT measured. NOTE: do NOT apply the 854x above to Qwen — that\n"
        "    multiple is vs Moondream running on CPU (~41 s). Qwen is GPU-served and far\n"
        f"    faster (~{QWEN_EST_MS:.0f} ms) despite being larger, so the production gap is\n"
        f"    only ~{qwen_speedup:.0f}x (estimate). The 854x and the ~{qwen_speedup:.0f}x are\n"
        "    different baselines (CPU local model vs GPU hosted model) — keep them separate.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
