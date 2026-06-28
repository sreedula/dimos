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

### Natural-language queries

`get_object_bbox` parses the description and disambiguates with the right strategy — all on the fast path, no VLM call:

```python
get_object_bbox(vl_model, image, "the leftmost person", detector=detector)  # spatial
get_object_bbox(vl_model, image, "the biggest chair",   detector=detector)  # largest/nearest
get_object_bbox(vl_model, image, "the red mug",         detector=detector)  # appearance (CLIP re-rank)

# Tracking: feed the previous frame's box so the same instance stays locked,
# instead of jumping to a higher-confidence different instance.
prev = get_object_bbox(vl_model, image, "person", detector=detector)
prev = get_object_bbox(vl_model, next_image, "person", detector=detector, prev_box=prev)
```

- **Spatial** — `leftmost / rightmost / topmost / bottommost / largest (biggest, nearest) / smallest / center`, resolved geometrically.
- **Appearance** — a multi-word phrase ("red mug") re-ranks same-class candidates by CLIP similarity; degrades to the top box if CLIP is unavailable.
- **Tracking** — `prev_box` selects the candidate most consistent with the last box (highest IoU, else nearest center).

The `navigation` skill feeds its last goal box back as `prev_box` automatically, so re-grounding the same goal across frames stays locked on one instance. The `person_follow` skill uses the same hint to **recover from tracking loss**: when EdgeTAM loses the target it re-grounds with YOLOE (hinted by the last box) and re-initializes the tracker — re-locking the *same* person after a brief occlusion instead of giving up.

The underlying primitives live in `dimos.navigation.visual.grounding` (`resolve_grounding`, `ground_candidates_with_yoloe`, `select_by_position`, `select_by_clip`, `select_nearest`). The detector memoizes the prompt, so grounding the same object across consecutive frames skips the text re-encode and pays only the detection cost.

## Benchmark

Measured with `bench_grounding.py` on the bundled `bus.jpg` / `zidane.jpg` samples. Hardware: Apple Silicon Mac, **CPU only** (no CUDA) — both YOLOE and the local VLM run on CPU, so the headline speedup is a same-hardware comparison.

**Speed — YOLOE vs a real local VLM (both measured)**

| Path | Latency | Source |
| --- | --- | --- |
| YOLOE fast-path (CPU) | **~47 ms** | measured, CPU steady-state (warmup excluded) |
| YOLOE fast-path (MPS) | **~24 ms** | measured, Apple-Silicon GPU |
| Moondream VLM (local, ~2B) | **~41,000 ms** | **measured**, CPU |
| Speedup | **~854×** | **measured**, same CPU |

The speedup is now a **measured** number, not an estimate. The detector auto-selects the fastest available backend — CUDA, then Apple-Silicon **MPS** (~2× faster than CPU here), then CPU — and falls back to CPU if a GPU op is unsupported, so it's a free win where the hardware allows. Two honest qualifiers below.

**Accuracy — agreement of YOLOE's box with two independent references**

| Reference | median IoU | Reading |
| --- | --- | --- |
| YOLO11 (standard detector) | **0.98** | strong agreement with a trusted detector |
| Moondream (~2B local VLM) | 0.17 | low — but Moondream is a weak grounder, so this reflects Moondream, not YOLOE |

The trustworthy accuracy signal is **0.98 vs YOLO11**. The low Moondream agreement is itself informative: a small local VLM is a poor grounder — part of why a fast, accurate detector path is worth having. (Multi-instance images like `zidane.jpg` also lower IoU because different models pick different *valid* people, not because anyone is wrong.)

**Aggregate accuracy vs HUMAN ground truth (COCO128)**

The two numbers above are on two images against another *detector*. `eval_grounding.py` broadens this to an aggregate over all 128 [COCO128](https://docs.ultralytics.com/datasets/detect/coco/) images, scoring YOLOE's grounded boxes against COCO's **human-annotated** labels for `person, car, truck, dog, chair` (CPU, the shipped conf=0.6). 77 images contained an eval class → **363 GT instances** (140 prominent, 223 tiny/background).

| Cut | GT instances | median IoU vs human GT | hit-rate @ IoU≥0.5 |
| --- | --- | --- | --- |
| **Prominent targets** (GT box ≥ 2% of frame) | 140 | **0.88** | **66%** |
| All GT instances | 363 | 0.00 | 34% |

On **prominent targets** — the objects a robot actually grounds and drives toward — YOLOE matches the human box at **median IoU 0.88** and localizes 66% of them well (IoU≥0.5), corroborating the 0.98-vs-YOLO11 signal against *real* ground truth over many images. The all-instances row collapses to 0.00 because COCO exhaustively annotates tiny/occluded background instances (223 of the 363) that the conf=0.6 fast-path deliberately does **not** fire on — that is a recall limit of the threshold, not a localization error: median IoU over the boxes YOLOE *does* return is still 0.87. Class-presence recall (a present class returned ≥1 box) is 69% (70/101). All CPU, still images only — an on-robot / simulator grounding run remains a separate manual follow-up.

That recall/precision trade-off is **tunable**: `build_yoloe_grounding_detector(confidence=0.25)` lowers the threshold to find smaller/occluded targets (on `bus.jpg`, 0.6 → 3 people vs 0.25 → 5), while the default 0.6 favors precision. Nothing changes for existing callers, who keep 0.6.

> **Honesty note — two different baselines, don't conflate them.**
> - Everything above (YOLOE ~47 ms, Moondream ~41 s, 854×, the IoUs) is **measured on this CPU**.
> - **854× is vs Moondream-on-CPU**, which is unusually slow. The *production* fallback is the hosted **Qwen2.5-VL-72B**, which is GPU-served and far faster (~2 s) despite being larger. So the production gap (YOLOE vs hosted Qwen) is only **~40×, and that one is an estimate** — there's no `ALIBABA_API_KEY` on the test machine to measure it. Never apply the 854× to Qwen.
> - The bus.jpg/zidane.jpg IoUs (0.98 vs YOLO11, 0.17 vs Moondream) are inter-detector agreement. The COCO128 table is the one accuracy result here measured against **human ground truth** (median IoU 0.87 on prominent targets); reproduce it with `eval_grounding.py`.
