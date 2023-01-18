import numpy as np
import pytest

from replay_buffer import FloatReplayBuffer


@pytest.fixture
def buffer_dict():
    return {
        "0": FloatReplayBuffer(priorities=np.array([1.2, 3.0, 4.0])),
        "1": FloatReplayBuffer(priorities=np.array([1.0, 5.0, 1.0])),
    }
