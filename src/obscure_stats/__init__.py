"""Collection of lesser-known statistical functions."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("obscure_stats")
except PackageNotFoundError:  # running from a source checkout that is not installed
    __version__ = "unknown"

__all__ = ["__version__"]
