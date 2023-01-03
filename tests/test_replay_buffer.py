import numpy as np

from replay_buffer import FloatReplayBuffer, TimeReplayBuffer


def test_time_sample_with_large_c():
    # Given a TimeReplayBuffer with 2 dates a few years apart
    priorities = np.array(
        [np.datetime64("2020-02-25T03:30"), np.datetime64("2022-08-25T23:30")]
    )
    buffer = TimeReplayBuffer(priority=priorities)

    # Check that sampling with large c always picks the earlier date
    assert buffer.sample(c=1_000) == 0


def test_get_and_set():
    # Given a FloatReplayBuffer with a few entries
    priorities = np.array([1.2, 3.0])
    buffer = FloatReplayBuffer(priority=priorities)

    # Check that we can get the provided values
    assert buffer.get_value(0) == 1.2
    assert buffer.get_value(1) == 3.0

    # Check that setting a value updates the get results
    buffer.set_value(0, 42.0)
    assert buffer.get_value(0) == 42.0
    assert buffer.get_value(1) == 3.0
