"""Head-to-head grounding benchmark: YOLOE fast-path vs Qwen-VLM slow-path.

WHAT THIS COMPARES
  - Fast path : dimos.navigation.visual.grounding.ground_with_yoloe  (open-vocab
                YOLOE detector, runs locally).
  - Slow path : dimos.navigation.visual.query.get_object_bbox_from_image, which
                grounds via the repo's real Qwen VL model
                (dimos.models.vl.create.create("qwen") -> qwen2.5-vl-72b-instruct,
                a HOSTED Alibaba DashScope API call).

METHODOLOGY (read before trusting any number)
  - Hardware: CPU-only Mac (Apple Silicon, arm64), no CUDA. YOLOE runs on CPU.
  - YOLOE latency is MEASURED: per query we do untimed warmup calls (model init +
    prompt encode), then time N steady-state calls and report the median. Warmup
    is excluded for both paths by construction.
  - Qwen latency is the honest weak spot on this host. Qwen is NOT a local model;
    it is a hosted 72B VLM reached over an OpenAI-compatible API that requires an
    ALIBABA_API_KEY. This script ACTUALLY ATTEMPTS one real Qwen call. If that
    call succeeds, its latency and bbox are MEASURED and used directly. If it
    fails (e.g. no API key on this machine), the script prints a
    MEASUREMENT-UNAVAILABLE note and falls back to a clearly-labeled ESTIMATE —
    it does NOT invent a measured number.
  - Qwen latency ESTIMATE basis (used only when the live call is unavailable):
    naive self-hosted Qwen2.5-VL-72B is reported at ~25-35 s/image on a single
    H100/A6000 GPU; a production hosted endpoint (DashScope) uses optimized
    serving and is typically a few seconds for one image + short JSON output. We
    use a deliberately CONSERVATIVE 2000 ms representative (range ~1.5-5 s) so the
    speedup is not overstated. Sources:
      https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/discussions/4
      https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-dashscope
  - Accuracy is MEASURED as IoU + center-distance against an independent
    closed-vocab reference detector (YOLO11), since the VLM box is unavailable
    here. This is inter-detector agreement, not human ground truth; when several
    instances match (e.g. multiple people in one frame) the two models may lock
    onto different valid boxes, lowering IoU without either being "wrong".
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import ultralytics

from dimos.models.vl.create import create
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.visual.grounding import ground_with_yoloe
from dimos.navigation.visual.query import get_object_bbox_from_image
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector, YoloePromptMode

# Conservative hosted-API estimate, used ONLY if the live Qwen call is unavailable.
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


def probe_qwen(vl_model, image) -> tuple[bool, str]:
    """Attempt one real Qwen call. Return (measured_ok, reason)."""
    try:
        t0 = time.perf_counter()
        _ = get_object_bbox_from_image(vl_model, image, "person")
        dt = (time.perf_counter() - t0) * 1000.0
        return True, f"live call OK ({dt:.0f} ms warmup)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


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

    # Build the REAL Qwen model and actually try it once (honest probe).
    vl_model = create("qwen")
    qwen_ok, qwen_reason = probe_qwen(vl_model, cases[0][1])
    print(f"qwen probe: model={vl_model.config.model_name}  measured={qwen_ok}  ({qwen_reason})")
    if not qwen_ok:
        print(
            "\n*** MEASUREMENT-UNAVAILABLE: live Qwen latency could not be measured on "
            "this host. ***\n"
            "    Reason: the hosted DashScope API needs ALIBABA_API_KEY, which is not set\n"
            "    here (env unset; default.env placeholder is empty). The endpoint itself is\n"
            "    reachable, so this is a credentials gap, not a compute limit. Qwen numbers\n"
            "    below are a CONSERVATIVE DOCUMENTED ESTIMATE (see file header), not measured.\n"
        )

    ref_model = build_reference_detector()

    rows = []
    for fname, img, q in cases:
        yolo_bbox, yolo_ms = time_yolo(detector, img, q)

        if qwen_ok:
            t0 = time.perf_counter()
            _ = get_object_bbox_from_image(vl_model, img, q)
            qwen_ms = (time.perf_counter() - t0) * 1000.0
        else:
            qwen_ms = QWEN_EST_MS

        speedup = qwen_ms / yolo_ms if yolo_ms > 0 else float("nan")

        # Measured accuracy signal: agreement with an independent closed-vocab
        # reference detector (YOLO11). Used because the VLM box is unavailable on
        # this host; it is an inter-detector agreement, not human ground truth.
        ref_bbox = reference_box(ref_model, img, q)
        if yolo_bbox is not None and ref_bbox is not None:
            ref_iou = iou(yolo_bbox, ref_bbox)
            ref_cd = center_dist(yolo_bbox, ref_bbox)
        else:
            ref_iou = ref_cd = None

        rows.append(
            {
                "q": q, "img": fname, "yolo_ms": yolo_ms, "qwen_ms": qwen_ms,
                "speedup": speedup, "yolo_bbox": yolo_bbox, "ref_bbox": ref_bbox,
                "ref_iou": ref_iou, "ref_cd": ref_cd,
            }
        )

    detector.stop()
    vl_model.stop()

    # ---- speed table: YOLOE (measured) vs Qwen-VLM (estimate here) ----
    print("\n" + "=" * 70)
    print(f"SPEED — YOLOE (measured) vs Qwen-VLM ({'measured' if qwen_ok else 'ESTIMATED'})")
    print("=" * 70)
    print(f"{'query':<9} {'image':<11} {'yolo_ms':>8} {'qwen_ms':>12} {'speedup':>10}")
    print("-" * 70)
    for r in rows:
        qcell = f"{r['qwen_ms']:7.0f}" if qwen_ok else f"~{r['qwen_ms']:.0f}(est)"
        scell = f"{r['speedup']:.0f}x" + ("" if qwen_ok else "(est)")
        print(f"{r['q']:<9} {r['img']:<11} {r['yolo_ms']:8.1f} {qcell:>12} {scell:>10}")
    print("-" * 70)

    # ---- accuracy table: YOLOE vs reference detector (measured) ----
    print("\n" + "=" * 70)
    print("ACCURACY — YOLOE grounding vs reference detector YOLO11 (measured agreement)")
    print("=" * 70)
    print(f"{'query':<9} {'image':<11} {'IoU':>6} {'center_px':>11}")
    print("-" * 70)
    for r in rows:
        icell = f"{r['ref_iou']:.2f}" if r["ref_iou"] is not None else "N/A"
        ccell = f"{r['ref_cd']:.0f}" if r["ref_cd"] is not None else "N/A"
        print(f"{r['q']:<9} {r['img']:<11} {icell:>6} {ccell:>11}")
    print("-" * 70)

    med_yolo = statistics.median([r["yolo_ms"] for r in rows])
    med_speedup = statistics.median([r["speedup"] for r in rows])
    ref_ious = [r["ref_iou"] for r in rows if r["ref_iou"] is not None]
    med_ref_iou = statistics.median(ref_ious) if ref_ious else None
    qlabel = "measured" if qwen_ok else "ESTIMATE (not measured — no API key, see header)"

    print("\n" + "=" * 70)
    print(f"median YOLOE latency : {med_yolo:.1f} ms (measured, CPU, steady-state)")
    print(f"median Qwen  latency : {QWEN_EST_MS if not qwen_ok else statistics.median([r['qwen_ms'] for r in rows]):.0f} ms [{qlabel}]")
    print(f"median speedup       : {med_speedup:.0f}x" + ("" if qwen_ok else "  [estimate-based]"))
    if med_ref_iou is not None:
        print(f"median IoU vs YOLO11 : {med_ref_iou:.2f} (measured; inter-detector agreement, not the VLM)")
    else:
        print("median IoU vs YOLO11 : N/A — no overlapping detections")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
