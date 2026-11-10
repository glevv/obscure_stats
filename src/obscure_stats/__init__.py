"""Collection of lesser-known statistical functions.

Modules:

- `obscure_stats.association`: measures of association between two variables.
- `obscure_stats.central_tendency`: measures of central tendency.
- `obscure_stats.dispersion`: measures of dispersion.
- `obscure_stats.kurtosis`: measures of kurtosis and tail weight.
- `obscure_stats.skewness`: measures of skewness.
- `obscure_stats.variation`: measures of qualitative variation (categorical data).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("obscure_stats")
except PackageNotFoundError:  # running from a source checkout that is not installed
    __version__ = "unknown"

__all__ = ["__version__"]
