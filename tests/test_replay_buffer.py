import numpy as np
import pytest

from replay_buffer import FloatReplayBuffer, TimeReplayBuffer


def test_time_sample_with_large_c():
    # Given a TimeReplayBuffer with 2 dates a few years apart
    priorities = np.array(
        [np.datetime64("2020-02-25T03:30"), np.datetime64("2022-08-25T23:30")]
    )
    buffer = TimeReplayBuffer(priorities=priorities)

    # Check that sampling with large c always picks the earlier date
    assert buffer.sample(c=1_000) == 0


def test_float_sample_with_large_c():
    # Given a FloatReplayBuffer with a few entries
    priorities = np.array([1.2, 3.0])
    buffer = FloatReplayBuffer(priorities=priorities)

    # Check that sampling with large c always picks the larger float
    assert buffer.sample(c=1_000) == 1


def test_get_and_set():
    # Given a FloatReplayBuffer with a few entries
    priorities = np.array([1.2, 3.0])
    buffer = FloatReplayBuffer(priorities=priorities)

    # Check that we can get the provided values
    assert buffer[0] == 1.2
    assert buffer[1] == 3.0

    # Check that setting a value updates the get results
    buffer[0] = 42.0
    assert buffer[0] == 42.0
    assert buffer[1] == 3.0


def test_buffer_len_with_append():
    # Given a FloatReplayBuffer with a few entries
    priorities = np.array([1.2, 3.0])
    buffer = FloatReplayBuffer(priorities=priorities)

    # Check that it has the expected length
    assert len(buffer) == 2

    # Check that appending an element changes the length
    buffer.append(42.0)
    assert len(buffer) == 3
    assert buffer[2] == 42.0


def test_get_index_out_of_range():
    # Given a FloatReplayBuffer with a few entries
    priorities = np.array([1.2, 3.0])
    buffer = FloatReplayBuffer(priorities=priorities)

    # Check that we get the expected error
    with pytest.raises(IndexError):
        buffer[2]
