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

"""Unit tests for PersonFollow's tracking-loss recovery (``_reacquire``).

Fully stubbed: a bare skill instance (no Module init), a fake EdgeTAM tracker,
and a monkeypatched ``get_object_bbox`` — so no models, robot, or LCM bus.
"""

from dimos.agents.skills import person_follow as pf
from dimos.agents.skills.person_follow import PersonFollowSkillContainer


def _bare_skill():
    """A PersonFollowSkillContainer with only the fields ``_reacquire`` touches."""
    skill = object.__new__(PersonFollowSkillContainer)
    skill._vl_model = object()  # stub; the patched get_object_bbox ignores it
    skill._grounding_detector = None
    skill._grounding_detector_built = True  # so _get_grounding_detector skips building
    return skill


class _Tracker:
    """Fake EdgeTAM tracker: init_track returns a list of `n_init` detections."""

    def __init__(self, n_init: int) -> None:
        self.n_init = n_init
        self.init_calls = 0

    def init_track(self, image, box, obj_id):
        self.init_calls += 1
        return list(range(self.n_init))


def test_reacquire_returns_box_and_reinits_tracker(monkeypatch) -> None:
    seen = {}

    def fake_get_object_bbox(vl_model, image, query, detector=None, prev_box=None):
        seen["prev_box"] = prev_box
        return (10.0, 20.0, 30.0, 40.0)

    monkeypatch.setattr(pf, "get_object_bbox", fake_get_object_bbox)
    skill = _bare_skill()
    tracker = _Tracker(n_init=1)

    box = skill._reacquire(tracker, "person", object(), prev_box=(9.0, 19.0, 29.0, 39.0))

    assert box == (10.0, 20.0, 30.0, 40.0)
    assert tracker.init_calls == 1  # tracker was re-initialized on the recovered box
    assert seen["prev_box"] == (9.0, 19.0, 29.0, 39.0)  # last box forwarded as the hint


def test_reacquire_returns_none_when_grounding_finds_nothing(monkeypatch) -> None:
    monkeypatch.setattr(pf, "get_object_bbox", lambda *a, **k: None)
    skill = _bare_skill()
    tracker = _Tracker(n_init=1)

    assert skill._reacquire(tracker, "person", object(), prev_box=None) is None
    assert tracker.init_calls == 0  # never tried to re-init without a box


def test_reacquire_returns_none_when_resegmentation_fails(monkeypatch) -> None:
    monkeypatch.setattr(pf, "get_object_bbox", lambda *a, **k: (1.0, 2.0, 3.0, 4.0))
    skill = _bare_skill()
    tracker = _Tracker(n_init=0)  # tracker fails to segment the recovered box

    assert skill._reacquire(tracker, "person", object(), prev_box=None) is None


def test_reacquire_swallows_grounding_exception(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("grounding blew up")

    monkeypatch.setattr(pf, "get_object_bbox", boom)
    skill = _bare_skill()

    # A grounding failure during recovery must not crash the follow loop.
    assert skill._reacquire(_Tracker(n_init=1), "person", object(), prev_box=None) is None
