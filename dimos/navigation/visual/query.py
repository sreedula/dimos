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

from dimos.models.qwen.bbox import BBox
from dimos.models.vl.base import VlModel
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.visual.grounding import ground_with_yoloe
from dimos.utils.generic import extract_json_from_llm_response
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def get_object_bbox(
    vl_model: VlModel,
    image: Image,
    object_description: str,
    detector=None,
) -> BBox | None:
    """Ground ``object_description`` to a bbox, YOLOE fast-path first, VLM fallback.

    When ``detector`` is provided, the open-vocab YOLOE detector is tried first
    (tens of ms). If it finds the object, that bbox is returned immediately. If
    it finds nothing, errors out, or no detector was given, this falls back to
    the slower Qwen-VLM path (``get_object_bbox_from_image``), so behavior is
    unchanged for callers that don't pass a detector.

    The fast path is best-effort: any exception from the detector (e.g. a model
    that failed to load, a missing text encoder) is swallowed and treated as a
    miss, so a broken fast path can never be worse than the VLM-only baseline.

    Args:
        vl_model: Vision-language model used for the fallback grounding.
        image: Image to ground against.
        object_description: Natural-language name of the object to locate.
        detector: Optional open-vocab detector for the fast path. If ``None``,
            only the VLM path is used.

    Returns:
        The object's ``(x1, y1, x2, y2)`` bbox, or ``None`` if neither path
        found it.
    """
    if detector is not None:
        try:
            bbox = ground_with_yoloe(detector, image, object_description)
        except Exception:
            logger.warning(
                "YOLOE fast-path grounding failed; falling back to the VLM.",
                exc_info=True,
            )
            bbox = None
        if bbox is not None:
            return bbox

    return get_object_bbox_from_image(vl_model, image, object_description)


def get_object_bbox_from_image(
    vl_model: VlModel, image: Image, object_description: str
) -> BBox | None:
    prompt = (
        f"Look at this image and find the '{object_description}'. "
        "Return ONLY a JSON object with format: {'name': 'object_name', 'bbox': [x1, y1, x2, y2]} "
        "where x1,y1 is the top-left and x2,y2 is the bottom-right corner of the bounding box. If not found, return None."
    )

    response = vl_model.query(image, prompt)

    result = extract_json_from_llm_response(response)
    if not result:
        return None

    try:
        ret = tuple(map(float, result["bbox"]))
        if len(ret) == 4:
            return ret
    except Exception:
        pass

    return None
