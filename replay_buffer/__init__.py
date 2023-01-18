import toml

from replay_buffer.multi_buffer import MultiBuffer  # NOQA: F401
from replay_buffer.replay_buffer import (  # NOQA: F401
    FloatReplayBuffer,
    TimeReplayBuffer,
)


def get_version() -> str:
    """Get version string from project's pyproject.toml

    Returns:
        str: Version string
    """
    pyproject = toml.load("pyproject.toml")
    return pyproject["tool"]["poetry"]["version"]


__version__ = get_version()
