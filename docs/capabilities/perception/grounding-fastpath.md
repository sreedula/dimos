# Grounding Fast-Path (YOLOE)

Grounding an object from a natural-language description ("find the person") used to go through a single path: a hosted Qwen vision-language model (VLM). That call is open-vocabulary and accurate, but it is a multi-second network round-trip to a 72B model — too slow to run every frame while a robot tracks a target.

## Solution

`get_object_bbox` puts an open-vocabulary **YOLOE** fast-path in front of the VLM:

1. If a detector is supplied, ground the description with YOLOE locally first.
2. On a hit, return that box immediately — tens of milliseconds.
3. On a miss (or when no detector is supplied), fall back to the existing VLM path.

YOLOE takes arbitrary text prompts ("person", "bus", "red mug") rather than a fixed class list, so it covers the same open-vocabulary queries the VLM does for common objects. Callers that pass no detector keep the exact previous VLM-only behavior, so the change is backward-compatible.

## Usage

```python
from dimos.models.vl.create import create
from dimos.navigation.visual.query import get_object_bbox
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector, YoloePromptMode

detector = Yoloe2DDetector(prompt_mode=YoloePromptMode.PROMPT, max_area_ratio=None)
vl_model = create("qwen")  # used only on a YOLOE miss

bbox = get_object_bbox(vl_model, image, "person", detector=detector)
# -> (x1, y1, x2, y2) or None
```

The underlying grounder is `dimos.navigation.visual.grounding.ground_with_yoloe`. It memoizes the prompt per detector, so grounding the same object across consecutive frames skips the text re-encode and pays only the detection cost.

## Benchmark

Measured with `bench_grounding.py` on the bundled `bus.jpg` / `zidane.jpg` samples.

**Speed**

| Path | Latency | Source |
| --- | --- | --- |
| YOLOE fast-path | **~48 ms** | measured, CPU steady-state (warmup excluded) |
| Qwen VLM | ~2000 ms | **estimate** — see note below |
| Speedup | **~42×** | **estimate-based** |

**Accuracy** — YOLOE's grounded box vs an independent reference detector (closed-vocab YOLO11):

| Query | IoU | Note |
| --- | --- | --- |
| person / bus.jpg | 0.98 | near-identical box |
| bus / bus.jpg | 0.98 | near-identical box |
| person / zidane.jpg | 0.31 | multiple people — the two detectors picked *different* valid persons |
| **median** | **0.98** | measured inter-detector agreement |

**Hardware:** Apple Silicon Mac, CPU only (no CUDA); YOLOE ran on CPU. A GPU would lower the YOLOE latency further.

> **Honesty note.** The **YOLOE latency (~48 ms) and the accuracy IoU are measured**; the **Qwen latency is not**. Qwen is a hosted DashScope API call needing `ALIBABA_API_KEY`, which was absent on the test machine, so its ~2000 ms is a conservative documented estimate (naive self-hosted 72B references are far higher) and the **~42× speedup is estimate-based**. Accuracy is measured against YOLO11 as a stand-in reference because the VLM box was unavailable, so it is inter-detector agreement, not human ground truth — and the multi-person zidane case shows the metric's limit (both boxes are valid), not a YOLOE error. With an API key set, `bench_grounding.py` also measures the VLM path live.
