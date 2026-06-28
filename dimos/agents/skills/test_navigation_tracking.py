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

"""Unit tests for NavigationSkillContainer's grounding tracking-continuity.

Fully stubbed: a bare skill instance (no Module init) with a monkeypatched
``get_object_bbox`` — no models, robot, or LCM bus.
"""

from dimos.agents.skills import navigation as nav
from dimos.agents.skills.navigation import NavigationSkillContainer


def _bare_skill():
    """A NavigationSkillContainer with only the fields the tracking path touches."""
    skill = object.__new__(NavigationSkillContainer)
    skill._latest_image = object()  # sentinel; the stubbed grounding ignores it
    skill._last_grounding_query = None
    skill._last_grounding_bbox = None
    skill._grounding_detector = None
    skill._grounding_detector_built = True  # so _get_grounding_detector skips building
    skill._vl_model = object()
    return skill


def test_navigation_tracking_continuity(monkeypatch) -> None:
    skill = _bare_skill()
    prev_boxes_seen: list = []
    ret = {"box": (1.0, 2.0, 3.0, 4.0)}

    def fake_get_object_bbox(vl, img, query, detector=None, prev_box=None):
        prev_boxes_seen.append(prev_box)
        return ret["box"]

    monkeypatch.setattr(nav, "get_object_bbox", fake_get_object_bbox)

    # Command 1: no prior box.
    assert skill._get_bbox_for_current_frame("person") == (1.0, 2.0, 3.0, 4.0)
    assert prev_boxes_seen[-1] is None

    # Command 2 (same query): the prior box is fed back as the hint.
    skill._get_bbox_for_current_frame("person")
    assert prev_boxes_seen[-1] == (1.0, 2.0, 3.0, 4.0)

    # Command 3: a miss returns None but must NOT drop the last box.
    ret["box"] = None
    assert skill._get_bbox_for_current_frame("person") is None

    # Command 4: still hinted by the last KNOWN box (survived the miss).
    ret["box"] = (5.0, 6.0, 7.0, 8.0)
    skill._get_bbox_for_current_frame("person")
    assert prev_boxes_seen[-1] == (1.0, 2.0, 3.0, 4.0)

    # A different query clears the continuity memory.
    skill._get_bbox_for_current_frame("chair")
    assert prev_boxes_seen[-1] is None


def test_navigation_get_bbox_returns_none_without_image() -> None:
    skill = _bare_skill()
    skill._latest_image = None
    assert skill._get_bbox_for_current_frame("person") is None
