from typing import Any, Dict, Optional

import numpy as np

from replay_buffer.replay_buffer import ReplayBufferMaster


class MultiBuffer(ReplayBufferMaster):
    def __init__(
        self, buffer_dict: Optional[Dict[str, ReplayBufferMaster]] = None
    ) -> None:
        """Initializes a MultiBuffer with the given buffer dict

        Args:
            buffer_dict (Optional[Dict[str,ReplayBufferMaster]], optional): An initial
            mapping from buffer name to replay buffer. Defaults to None.
        """
        if buffer_dict is None:
            self.buffer_dict = {}
        else:
            buffer_lengths = {len(buffer) for buffer in buffer_dict.values()}
            if len(buffer_lengths) > 1:
                raise ValueError(
                    "All input buffers must be of same length, "
                    f"but got buffers with lengths {buffer_lengths}"
                )
            self.buffer_dict = buffer_dict

    def append(self, values: Dict[str, Any]) -> None:
        # Check that the keys match those currently in the buffer dict
        if set(values.keys()) != set(self.buffer_dict.keys()):
            raise ValueError(
                "Keys do not match. Got: "
                f"{set(values.keys())}. Expected: {set(self.buffer_dict.keys())}"
            )

        # Append each item in its respective buffer
        for key, value in values.items():
            self.buffer_dict[key].append(value)

    def __len__(self) -> int:
        """Get the current length of the multi buffer

        Returns:
            int: _description_
        """
        for _, replay_buffer in self.buffer_dict.items():
            return len(replay_buffer)
        return 0

    def __getitem__(self, buffer_name: str) -> Any:
        return self.buffer_dict[buffer_name]

    def __setitem__(self, buffer_name: str, value: Any) -> None:
        self.buffer_dict[buffer_name] = value

    def distr(
        self, c: float = 1.0, weights: Optional[Dict[str, float]] = None
    ) -> np.array:
        """Get the current probability distribution over samples by taking a weighted
        average of the distributions from each replay buffer

        Args:
            c (float, optional): Exponent to control the sharpness of the distribution
            over samples. A value of 0 flattens the distribution into a uniform
            distribution, a value of infinity shaprens the distribution into a one-hot
            distribution, and a value of 1 leaves the distribution unchanged. Defaults
            to 1.0.

        Returns:
            np.array: The probability distribution over samples in the replay buffer
        """
        # Preprocess the weights dict
        n = len(self.buffer_dict)
        if weights is None:
            weights = {key: 1 / n for key in self.buffer_dict.keys()}
        else:
            if set(weights.keys()) != set(self.buffer_dict.keys()):
                raise ValueError(
                    "weights keys do not match. Got: "
                    f"{set(weights.keys())}. Expected: {set(self.buffer_dict.keys())}"
                )
            weight_sum = sum(weight for _, weight in weights.items())
            weights = {key: weight / weight_sum for key, weight in weights.items()}

        # Compute the distribution
        random_key = next(iter(self.buffer_dict.keys()))
        out = np.zeros_like(self.buffer_dict[random_key])
        for key, buffer in self.buffer_dict.items():
            out += weights[key] * buffer.distr(c=c)
        return out

    def sample(self, c: float = 1.0, weights: Optional[Dict[str, float]] = None) -> int:
        """Get the index of the "next" element in the multi buffer based on the
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
        distr = self.distr(c=c, weights=weights)
        return np.random.choice(len(distr), p=distr)
