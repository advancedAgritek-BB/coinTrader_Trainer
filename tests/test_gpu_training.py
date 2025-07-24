"""Placeholder for GPU training tests.

These tests require GPU hardware and are skipped in most environments.
"""

import pytest

pytest.skip(
    "Skipping GPU training tests; hardware not available",
    allow_module_level=True,
)
