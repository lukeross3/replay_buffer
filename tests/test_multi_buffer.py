import pytest

from replay_buffer import MultiBuffer


def test_different_buffer_lengths(buffer_dict):
    # Given a buffer dict with different buffer lengths
    buffer_dict["0"].append(1.0)

    # Check that initializing a MultiBuffer raises a ValueError
    with pytest.raises(ValueError):
        MultiBuffer(buffer_dict=buffer_dict)


def test_float_sample_with_large_c_and_weight(buffer_dict):
    # Given a MultiBuffer
    multi_buffer = MultiBuffer(buffer_dict=buffer_dict)

    # Check that sampling with large c and all the weight on a single buffer always
    # samples the highest priority element from that buffer
    assert multi_buffer.sample(c=1_000, weights={"0": 0, "1": 1}) == 1


def test_nested_get_and_set(buffer_dict):
    # Given a MultiBuffer
    multi_buffer = MultiBuffer(buffer_dict=buffer_dict)

    # Check that we can get the provided values
    assert multi_buffer["0"][0] == 1.2
    assert multi_buffer["1"][0] == 1.0

    # Check that setting a value updates the get results
    multi_buffer["1"][0] = 42.0
    assert multi_buffer["0"][0] == 1.2
    assert multi_buffer["1"][0] == 42.0


def test_simple_get_and_set(buffer_dict):
    # Given an empty MultiBuffer
    multi_buffer = MultiBuffer()

    # Check that getting raises a KeyError
    with pytest.raises(KeyError):
        multi_buffer["0"]

    # Set the value
    multi_buffer["0"] = buffer_dict["0"]

    # Check that we can now get the value
    assert multi_buffer["0"][0] == 1.2
