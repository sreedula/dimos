"""YOLOE grounding fast-path demo + latency self-test for DimOS.

The grounder itself now lives in the package at
``dimos.navigation.visual.grounding.ground_with_yoloe`` (single source of truth,
also used by ``dimos.navigation.visual.query.get_object_bbox``). This script
re-exports it and exercises it headless against the ultralytics ``bus.jpg``
sample to report steady-state latency.
"""

from __future__ import annotations

from dimos.navigation.visual.grounding import ground_with_yoloe

__all__ = ["ground_with_yoloe"]


if __name__ == "__main__":
    from pathlib import Path
    import time

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
