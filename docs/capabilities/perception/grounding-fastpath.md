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

Measured with `bench_grounding.py` on the bundled `bus.jpg` / `zidane.jpg` samples. Hardware: Apple Silicon Mac, **CPU only** (no CUDA) — both YOLOE and the local VLM run on CPU, so the headline speedup is a same-hardware comparison.

**Speed — YOLOE vs a real local VLM (both measured)**

| Path | Latency | Source |
| --- | --- | --- |
| YOLOE fast-path | **~47 ms** | measured, CPU steady-state (warmup excluded) |
| Moondream VLM (local, ~2B) | **~41,000 ms** | **measured**, CPU |
| Speedup | **~854×** | **measured**, same CPU |

The speedup is now a **measured** number, not an estimate. Two honest qualifiers below.

**Accuracy — agreement of YOLOE's box with two independent references**

| Reference | median IoU | Reading |
| --- | --- | --- |
| YOLO11 (standard detector) | **0.98** | strong agreement with a trusted detector |
| Moondream (~2B local VLM) | 0.17 | low — but Moondream is a weak grounder, so this reflects Moondream, not YOLOE |

The trustworthy accuracy signal is **0.98 vs YOLO11**. The low Moondream agreement is itself informative: a small local VLM is a poor grounder — part of why a fast, accurate detector path is worth having. (Multi-instance images like `zidane.jpg` also lower IoU because different models pick different *valid* people, not because anyone is wrong.)

> **Honesty note — two different baselines, don't conflate them.**
> - Everything above (YOLOE ~47 ms, Moondream ~41 s, 854×, the IoUs) is **measured on this CPU**.
> - **854× is vs Moondream-on-CPU**, which is unusually slow. The *production* fallback is the hosted **Qwen2.5-VL-72B**, which is GPU-served and far faster (~2 s) despite being larger. So the production gap (YOLOE vs hosted Qwen) is only **~40×, and that one is an estimate** — there's no `ALIBABA_API_KEY` on the test machine to measure it. Never apply the 854× to Qwen.
> - Accuracy here is inter-detector agreement, not human ground truth.
