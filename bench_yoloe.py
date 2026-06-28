"""Benchmark the inference latency of DimOS's open-vocabulary YOLOE detector.

Runs fully headless against a static sample image (ultralytics' bus.jpg, which
contains people and a bus). The first detector call includes one-time model
warmup, so process_image is run 5 times and every timing is printed to make the
warmup-vs-steady-state difference visible.
"""

from pathlib import Path
import time

import ultralytics

from dimos.msgs.sensor_msgs.Image import Image
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector, YoloePromptMode


def find_sample_image() -> Path:
    """Return ultralytics' bundled bus.jpg, downloading it if not present."""
    assets = Path(ultralytics.__file__).parent / "assets"
    bus = assets / "bus.jpg"
    if not bus.exists():
        import urllib.request

        assets.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ultralytics/ultralytics/raw/main/ultralytics/assets/bus.jpg"
        print(f"bus.jpg not bundled; downloading from {url}")
        urllib.request.urlretrieve(url, bus)
    return bus


def main() -> None:
    sample_path = find_sample_image()
    print(f"sample image: {sample_path}")
    img = Image.from_file(str(sample_path))
    print(f"image size: {img.width}x{img.height}")

    # Open-vocabulary detector with a TEXT target. bus.jpg contains both.
    detector = Yoloe2DDetector(prompt_mode=YoloePromptMode.PROMPT)
    detector.set_prompts(text=["person", "bus"])

    # Run 5 times. Run 0 includes warmup; runs 1-4 are steady-state.
    timings_ms = []
    result = None
    for i in range(5):
        t0 = time.perf_counter()
        result = detector.process_image(img)
        dt = (time.perf_counter() - t0) * 1000.0
        timings_ms.append(dt)
        print(f"run {i}: {dt:7.1f} ms, {len(result.detections)} detections")

    print("\ndetections (last run):")
    for d in result.detections:
        x1, y1, x2, y2 = (round(float(v)) for v in d.bbox)
        print(f"  {d.name:<12} conf={d.confidence:.2f}  bbox=({x1}, {y1}, {x2}, {y2})")

    steady = timings_ms[1:]
    steady_avg = sum(steady) / len(steady)
    names = sorted({d.name for d in result.detections})
    detected = ", ".join(names) if names else "nothing"
    print(
        f"\nSUMMARY: steady-state {steady_avg:.1f} ms "
        f"(runs 1-4: {min(steady):.1f}-{max(steady):.1f} ms); detected: {detected}"
    )

    detector.stop()


if __name__ == "__main__":
    main()
