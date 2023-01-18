from random import randint
from typing import Any, List, Optional

import numpy as np

from replay_buffer import ReplayBuffer


class Flashcards:
    def __init__(
        self, seen: List, unseen: List, replay_buffer: Optional[ReplayBuffer] = None
    ) -> None:
        self.seen = seen
        self.unseen = unseen
        self.replay_buffer = replay_buffer
        if replay_buffer is None:
            self.replay_buffer = ReplayBuffer(size=len(seen))

    def new_card(self) -> Any:
        """Pop a random element from the unseen list in constant time

        Returns:
            Any: A random element from the unseen list
        """
        i = randint(0, len(self.unseen))
        self.unseen[-1], self.unseen[i] = self.unseen[i], self.unseen[-1]
        return self.unseen.pop()

    def old_card(self, c: float = 1.0) -> Any:
        """Get a card from the seen list according to the replay buffer policy

        Returns:
            Any: An element from the seen list
        """
        i = self.replay_buffer.sample(c=c)
        return self.seen[i]

    def next_card(self) -> None:
        # TODO: how to know when to get old vs new? Maybe use distribution or
        # just raw priority values
        pass

    def add_to_seen(
        self, card: Any, time: np.datetime64 = np.datetime64("now")
    ) -> None:
        self.seen.append(card)
        self.replay_buffer.append(time=time)
