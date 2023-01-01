import toml


def get_version() -> str:
    """Get version string from project's pyproject.toml

    Returns:
        str: Version string
    """
    pyproject = toml.load("pyproject.toml")
    return pyproject["tool"]["poetry"]["version"]


__version__ = get_version()
