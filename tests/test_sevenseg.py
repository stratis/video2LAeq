import pytest

from synth import render_frame
from video2laeq.roi import find_lcd
from video2laeq.sevenseg import read_panel


@pytest.mark.parametrize("value", [47.4, 60.1, 55.9, 30.0, 99.9, 82.6, 71.3, 128.3])
def test_reads_synthetic_panel(value):
    frame = render_frame(value, seed=int(value * 10), body_w=200 if value >= 100 else 140)
    rect = find_lcd(frame)
    assert rect is not None
    reading = read_panel(rect.crop(frame))
    assert reading.value == pytest.approx(value), reading


def test_auto_decimal_point():
    frame = render_frame(64.2, body_w=200)
    reading = read_panel(find_lcd(frame).crop(frame), decimals=None)
    assert reading.value == pytest.approx(64.2), reading


def test_aggregate():
    from video2laeq.metrics import aggregate
    assert aggregate([]) is None
    assert aggregate([55.1]) == 55.1
    assert aggregate([55.1, 55.3, 95.7, 55.2]) == pytest.approx(55.2)
    assert aggregate([55.1, 95.7]) is None


def test_no_lcd_in_plain_frame():
    import numpy as np
    assert find_lcd(np.zeros((640, 360, 3), np.uint8)) is None
