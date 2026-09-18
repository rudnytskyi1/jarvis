"""server/vision.py: how a grounding answer becomes a click (v1.7.1).

Qwen2.5-VL answers in absolute pixels of the image it was given, whatever the
prompt asks for. Measured on a real YouTube screenshot whose search box sits at
(730, 143) of 1600x900, it replied y=142 at 1600x900 and y=92 at 1024x576 - the
number scales with the image, which is what pixels do. The old code divided
those by 1000 whenever they were under 1000, so "click the search box" landed on
the microphone icon, and at 1024 wide it landed in the browser address bar,
turning a typed query into a Google search.
"""
import pytest

from server.vision import LOCATE_PROMPT_TEMPLATE, to_normalized


def test_a_real_answer_lands_on_the_search_box():
    # The exact reply measured from the model for the YouTube search box.
    x, y, space = to_normalized(720, 143, 1600, 900)
    assert space == "pixels"
    assert x == pytest.approx(0.45, abs=0.02)   # true centre 730/1600 = 0.456
    assert y == pytest.approx(0.159, abs=0.02)


def test_the_old_grid_reading_would_have_missed():
    # What the previous code did with the same reply: divide by 1000.
    wrong_x = 720 / 1000
    assert abs(wrong_x - 0.456) > 0.25  # a quarter of the screen away


def test_pixels_scale_with_the_image_size():
    # Same element, two image sizes, each answered in that image's pixels.
    big = to_normalized(720, 143, 1600, 900)
    small = to_normalized(461, 92, 1024, 576)
    assert big[0] == pytest.approx(small[0], abs=0.02)
    assert big[1] == pytest.approx(small[1], abs=0.02)


def test_a_zero_to_one_fraction_is_recognised():
    x, y, space = to_normalized(0.45, 0.16, 1600, 900)
    assert space == "0-1 fraction"
    assert x == pytest.approx(0.45)
    assert y == pytest.approx(0.16)


def test_out_of_frame_is_clamped_to_the_edge():
    x, y, _ = to_normalized(5000, -40, 1600, 900)
    assert x == 1.0
    assert y == 0.0


def test_a_degenerate_image_size_does_not_divide_by_zero():
    x, y, _ = to_normalized(10, 10, 0, 0)
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_the_prompt_asks_for_pixels_not_a_grid():
    filled = LOCATE_PROMPT_TEMPLATE.format(width=1600, height=900, target="x", grid=1000)
    assert "PIXEL" in filled
    assert "normalized grid" not in filled
    assert "0-1600" in filled and "0-900" in filled
