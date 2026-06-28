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

"""Fast, stubbed unit tests for Yoloe2DDetector behaviour that needs no weights.

These bypass __init__ (which loads the real model) via object.__new__, so they
run without Git-LFS data or a GPU. End-to-end detector behaviour is covered by
the self_hosted tests in test_bbox_detectors.py.
"""

from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector


class _StubModel:
    """Minimal stand-in for the ultralytics model: only `predictor` is read."""

    def __init__(self) -> None:
        self.predictor = None


def test_stop_clears_text_pe_cache() -> None:
    detector = object.__new__(Yoloe2DDetector)
    detector.model = _StubModel()
    detector._text_pe_cache = {("person",): object(), ("bus",): object()}

    detector.stop()

    # The cached embeddings (tensors, possibly on GPU) are released on cleanup.
    assert detector._text_pe_cache == {}


def test_stop_is_safe_without_a_cache_attribute() -> None:
    # A detector constructed in some odd state (no cache yet) must not crash.
    detector = object.__new__(Yoloe2DDetector)
    detector.model = _StubModel()

    detector.stop()  # must not raise
