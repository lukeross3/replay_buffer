import toml


def get_version():
    pyproject = toml.load("pyproject.toml")
    return pyproject["tool"]["poetry"]["version"]

__version__ = get_version()