"""Tests of package metadata."""

import obscure_stats


def test_version_is_a_string() -> None:
    """Test that __version__ exists (it is "unknown" in an uninstalled checkout)."""
    if not isinstance(obscure_stats.__version__, str) or not obscure_stats.__version__:
        msg = f"Unexpected __version__: {obscure_stats.__version__!r}"
        raise ValueError(msg)
