import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.timing import timed


def test_timed_logs_when_flag(caplog):
    @timed
    def sample(x):
        return x + 1

    with caplog.at_level(logging.INFO):
        result = sample(1, log_time=True)
    assert result == 2
    assert any("sample took" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_timed_logs_async(caplog):
    @timed
    async def sample_async(x):
        return x + 1

    with caplog.at_level(logging.INFO):
        result = await sample_async(1, log_time=True)
    assert result == 2
    assert any("sample_async took" in r.message for r in caplog.records)
