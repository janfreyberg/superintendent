import time

from superintendent.controls import Timer


def test_that_timer_stores_the_time():
    timer = Timer()
    with timer:
        time.sleep(0.01)
    assert timer._time >= 0.01


def test_that_timer_compares_correctly():
    timer = Timer()
    with timer:
        time.sleep(0.01)

    assert timer >= 0.01
    assert timer < 0.1
    assert timer == timer._time
