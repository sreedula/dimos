"""YOLOE grounding fast-path for DimOS.

`ground_with_yoloe` turns a natural-language object description into a single
bounding box using the open-vocabulary YOLOE detector. It is the fast
counterpart to the slow Qwen-VLM grounding in
`dimos/navigation/visual/query.py::get_object_bbox_from_image`, and returns the
same `(x1, y1, x2, y2)` float-tuple convention so it can drop in with minimal
changes.

The function takes an already-constructed detector and image — it does NOT build
them itself. It memoizes the last text prompt per detector: re-encoding the
prompt (a CLIP text-encoder pass) dominates per-call latency, so when the
description is unchanged from the detector's previous call — the common
navigation case of grounding the same object across consecutive video frames —
that cost is skipped and only the detection runs.
"""

from __future__ import annotations


def ground_with_yoloe(detector, image, description: str) -> tuple[float, float, float, float] | None:
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


if __name__ == "__main__":
    import time
    from pathlib import Path

    import ultralytics

    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector, YoloePromptMode

    # Same sample image bench_yoloe.py uses: people + a bus, ships with ultralytics.
    sample_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    img = Image.from_file(str(sample_path))
    print(f"sample image: {sample_path} ({img.width}x{img.height})")

    # max_area_ratio=None disables the >30%-of-frame filter so the large bus survives.
    detector = Yoloe2DDetector(prompt_mode=YoloePromptMode.PROMPT, max_area_ratio=None)

    # One untimed call to warm up the model (first inference pays a ~3s init).
    ground_with_yoloe(detector, img, "person")

    queries = ["person", "bus", "stop sign", "banana"]
    results: list[tuple[str, tuple[float, float, float, float] | None, float]] = []
    for q in queries:
        # Warmup ground for this query: encodes & sets the prompt (one-time cost,
        # excluded from the steady-state timing below).
        ground_with_yoloe(detector, img, q)

        # Steady-state ground: prompt already cached, so this is detection-only —
        # the per-frame cost a navigator pays once locked onto a target.
        t0 = time.perf_counter()
        bbox = ground_with_yoloe(detector, img, q)
        dt = (time.perf_counter() - t0) * 1000.0

        results.append((q, bbox, dt))
        bbox_str = "None" if bbox is None else "(" + ", ".join(f"{v:.0f}" for v in bbox) + ")"
        print(f"  query={q!r:<12} bbox={bbox_str:<26} latency={dt:6.1f} ms")

    detector.stop()

    summary = "  |  ".join(
        f"{q}: {'None' if b is None else '(' + ','.join(f'{v:.0f}' for v in b) + ')'} @ {dt:.0f}ms"
        for q, b, dt in results
    )
    print(f"\nSUMMARY: {summary}")
