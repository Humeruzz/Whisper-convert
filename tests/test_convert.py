import pytest
from unittest.mock import MagicMock, patch

from convert import format_timestamp, write_transcript


def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00"


def test_format_timestamp_seconds_only():
    assert format_timestamp(45.9) == "00:00:45"


def test_format_timestamp_minutes_and_seconds():
    assert format_timestamp(62.5) == "00:01:02"


def test_format_timestamp_hours_minutes_seconds():
    assert format_timestamp(3723.0) == "01:02:03"


def test_write_transcript_creates_file(tmp_path):
    out = tmp_path / "out.txt"
    write_transcript([(2.0, "Hello world"), (65.0, "Goodbye")], str(out))
    assert out.exists()


def test_write_transcript_format(tmp_path):
    out = tmp_path / "out.txt"
    write_transcript([(2.0, "Hello world"), (65.0, "Goodbye")], str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[00:00:02] Hello world"
    assert lines[1] == "[00:01:05] Goodbye"


def test_write_transcript_empty(tmp_path):
    out = tmp_path / "empty.txt"
    write_transcript([], str(out))
    assert out.read_text(encoding="utf-8") == ""
