"""parse_point: the grounding reply must survive every reply shape."""
import pytest

from server.vision import parse_point


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"x": 640, "y": 360}', (640.0, 360.0)),
        ('Sure! The point is {"x": 12, "y": 900}.', (12.0, 900.0)),
        ('```json\n{"x": 100, "y": 200}\n```', (100.0, 200.0)),
        ('{"x": "640px", "y": "360"}', (640.0, 360.0)),
        ("x=300, y=400", (300.0, 400.0)),
        ("not found", None),
        ("", None),
        (None, None),
        ("The element is not visible anywhere.", None),
    ],
)
def test_parse_point(reply, expected):
    assert parse_point(reply) == expected


def test_parse_point_prefers_json_over_stray_numbers():
    # The 1024 in the prose must not win over the JSON object.
    assert parse_point('The image is 1024 wide. {"x": 5, "y": 6}') == (5.0, 6.0)
