from abc import ABC
from typing import Any, Optional

import numpy as np


class ReplayBufferMaster(ABC):
    dtype = float

    def __init__(self, priorities: Optional[np.array] = None) -> None:
        """Initialize a replay buffer with the given priority values. If no priority
        values provided, the replay buffer is initialized as empty.

        Args:
            priorities (Optional[np.array], optional): An array of priority values.
            Defaults to None.
        """
        self.priorities = (
            np.empty(0, dtype=self.dtype) if priorities is None else priorities
        )

    def append(self, value: Any) -> None:
        """Add a new element to the replay buffer with the given priority value. May be
        a time, a float, etc.

        Args:
            value (Any): Value to give the new item's priority
        """
        self.priorities = np.append(self.priorities, value)

    def __len__(self) -> int:
        """Get the current length of the replay buffer

        Returns:
            int: _description_
        """
        return len(self.priorities)

    def __getitem__(self, index: int) -> Any:
        """Get a replay buffer item's priority. May be a time, a float, etc.

        Args:
            index (int): Index of item to update
            value (Any): Value to give item's priority
        """
        return self.priorities[index]

    def __setitem__(self, index: int, value: Any) -> None:
        """Update a replay buffer item's priority. May be a time, a float, etc.

        Args:
            index (int): Index of item to update
            value (Any): Value to give item's priority
        """
        self.priorities[index] = value

    def sample(self, c: float = 1.0) -> int:
        """Get the index of the "next" element in the replay buffer based on the
        current priority values.

        Args:
            c (float, optional): Exponent to control the sharpness of the distribution
            over samples. A value of 0 flattens the distribution into a uniform
            distribution, a value of infinity shaprens the distribution into a one-hot
            distribution, and a value of 1 leaves the distribution unchanged. Defaults
            to 1.0.

        Returns:
            int: The index of the element sampled from the replay buffer
        """


class TimeReplayBuffer(ReplayBufferMaster):
    """A replay buffer which prioritizes items based on the elapsed time since each
    item was last seen. The priority of an item is simply a datetime representing the
    point in time that item was last seen. During sampling, the elapsed time is
    calculated for each element, normalized to sum to 1, and taken as the distribution
    over items.
    """

    dtype = np.datetime64

    def sample(self, c: float = 1.0) -> int:
        """Get the index of the "next" element in the replay buffer based on the
        current priority values.

        Args:
            c (float, optional): Exponent to control the sharpness of the distribution
            over samples. A value of 0 flattens the distribution into a uniform
            distribution, a value of infinity shaprens the distribution into a one-hot
            distribution, and a value of 1 leaves the distribution unchanged. Defaults
            to 1.0.

        Returns:
            int: The index of the element sampled from the replay buffer
        """
        elapsed = np.datetime64("now") - self.priorities
        elapsed_seconds = elapsed / np.timedelta64(1, "s")
        normalized = elapsed_seconds / np.sum(elapsed_seconds)
        exponentiated = normalized**c
        distr = exponentiated / np.sum(exponentiated)
        return np.random.choice(len(distr), p=distr)


class FloatReplayBuffer(ReplayBufferMaster):
    """A replay buffer which prioritizes items based on a float value from 0 to 1. The
    items with values near 1 are prioritized over those near 0.
    """

    dtype = np.float64

    def sample(self, c: float = 1.0) -> int:
        """Get the index of the "next" element in the replay buffer based on the
        current priority values.

        Args:
            c (float, optional): Exponent to control the sharpness of the distribution
            over samples. A value of 0 flattens the distribution into a uniform
            distribution, a value of infinity shaprens the distribution into a one-hot
            distribution, and a value of 1 leaves the distribution unchanged. Defaults
            to 1.0.

        Returns:
            int: The index of the element sampled from the replay buffer
        """
        normalized = self.priorities / np.sum(self.priorities)
        exponentiated = normalized**c
        distr = exponentiated / np.sum(exponentiated)
        return np.random.choice(len(distr), p=distr)
