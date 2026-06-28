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

from uuid import UUID

import pytest

from dimos.utils.generic import extract_json_from_llm_response, short_id


@pytest.mark.parametrize(
    "response,expected",
    [
        ('{"name": "cup", "bbox": [1, 2, 3, 4]}', {"name": "cup", "bbox": [1, 2, 3, 4]}),
        ('the box is {"name": "cup"} ok', {"name": "cup"}),
        ("sorry, not found", None),  # no JSON
        ("", None),  # empty
        (None, None),  # model returned nothing (timeout/error) -> no crash
        (123, None),  # non-string -> no crash
    ],
)
def test_extract_json_from_llm_response(response, expected) -> None:
    assert extract_json_from_llm_response(response) == expected


def test_short_id_hello_world() -> None:
    assert short_id("HelloWorld") == "6GgJmzi1KYf4iaHVxk"


def test_short_id_uuid_one(mocker) -> None:
    mocker.patch("uuid.uuid4", return_value=UUID("11111111-1111-1111-1111-111111111111"))
    assert short_id() == "wcFtOGNXQnQFZ8QRh1"


def test_short_id_uuid_zero(mocker) -> None:
    mocker.patch("uuid.uuid4", return_value=UUID("00000000-0000-0000-0000-000000000000"))
    assert short_id() == "000000000000000000"
