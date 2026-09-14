"""observability.RingBufferHandler + /logs, /logs/stats."""
from __future__ import annotations

import logging

from observability import RingBufferHandler


def test_ring_buffer_captures_and_filters():
    h = RingBufferHandler(capacity=10)
    logger = logging.getLogger("dagkb.test.ring")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(h)
    logger.propagate = False
    try:
        logger.info("hello world")
        logger.warning("careful now")
        logger.debug("quiet detail")
    finally:
        logger.removeHandler(h)

    all_rows = h.snapshot()
    assert len(all_rows) == 3
    assert h.snapshot(level="WARNING") == [r for r in all_rows if r["level"] == "WARNING"]
    assert len(h.snapshot(contains="careful")) == 1
    assert len(h.snapshot(logger_prefix="dagkb.test")) == 3


def test_ring_buffer_is_bounded():
    h = RingBufferHandler(capacity=3)
    logger = logging.getLogger("dagkb.test.ring2")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(h)
    logger.propagate = False
    try:
        for i in range(10):
            logger.info("line %d", i)
    finally:
        logger.removeHandler(h)
    rows = h.snapshot(limit=100)
    assert len(rows) == 3
    assert rows[-1]["message"] == "line 9"
