import pytest
from unittest.mock import MagicMock, patch

from convert import format_timestamp


def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00"


def test_format_timestamp_seconds_only():
    assert format_timestamp(45.9) == "00:00:45"


def test_format_timestamp_minutes_and_seconds():
    assert format_timestamp(62.5) == "00:01:02"


def test_format_timestamp_hours_minutes_seconds():
    assert format_timestamp(3723.0) == "01:02:03"
