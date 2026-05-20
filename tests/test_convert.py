import pytest
from unittest.mock import MagicMock, patch

from convert import format_timestamp, write_transcript, transcribe, load_model, main


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


def test_transcribe_returns_segments():
    seg1 = MagicMock()
    seg1.start = 1.0
    seg1.text = "  Hello  "
    seg2 = MagicMock()
    seg2.start = 5.5
    seg2.text = "World"

    model = MagicMock()
    model.transcribe.return_value = ([seg1, seg2], MagicMock())

    result = transcribe(model, "fake.mp4")

    assert result == [(1.0, "Hello"), (5.5, "World")]


def test_transcribe_strips_whitespace():
    seg = MagicMock()
    seg.start = 0.0
    seg.text = "   padded text   "

    model = MagicMock()
    model.transcribe.return_value = ([seg], MagicMock())

    result = transcribe(model, "fake.mp4")

    assert result == [(0.0, "padded text")]


def test_transcribe_calls_model_with_correct_params():
    model = MagicMock()
    model.transcribe.return_value = ([], MagicMock())

    transcribe(model, "video.mp4")

    call_kwargs = model.transcribe.call_args
    assert call_kwargs[0][0] == "video.mp4"
    assert call_kwargs[1]["beam_size"] == 5
    assert call_kwargs[1]["vad_filter"] is True
    assert call_kwargs[1]["vad_parameters"] == {"min_silence_duration_ms": 500}


def test_transcribe_empty_file():
    model = MagicMock()
    model.transcribe.return_value = ([], MagicMock())

    result = transcribe(model, "silent.mp4")

    assert result == []


def test_load_model_returns_whisper_instance():
    with patch("convert.WhisperModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance

        result = load_model()

        assert result is mock_instance


def test_load_model_uses_env_config():
    with patch("convert.WhisperModel") as MockModel:
        with patch("convert.MODEL_SIZE", "base"):
            with patch("convert.COMPUTE_TYPE", "float16"):
                load_model()
                MockModel.assert_called_once_with("base", device="cpu", compute_type="float16")


def test_main_file_not_found(capsys):
    with patch("sys.argv", ["convert.py", "nonexistent.mp4"]):
        result = main()

    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_main_default_output_path(tmp_path):
    fake_mp4 = tmp_path / "lecture.mp4"
    fake_mp4.write_bytes(b"fake-data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())

    with patch("sys.argv", ["convert.py", str(fake_mp4)]):
        with patch("convert.load_model", return_value=mock_model):
            result = main()

    assert result == 0
    assert (tmp_path / "lecture.txt").exists()


def test_main_custom_output_path(tmp_path):
    fake_mp4 = tmp_path / "lecture.mp4"
    fake_mp4.write_bytes(b"fake-data")
    custom_out = tmp_path / "notes.txt"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())

    with patch("sys.argv", ["convert.py", str(fake_mp4), "-o", str(custom_out)]):
        with patch("convert.load_model", return_value=mock_model):
            result = main()

    assert result == 0
    assert custom_out.exists()


def test_main_writes_segments_to_file(tmp_path):
    fake_mp4 = tmp_path / "video.mp4"
    fake_mp4.write_bytes(b"fake-data")

    seg1 = MagicMock(); seg1.start = 3.0; seg1.text = "Hello"
    seg2 = MagicMock(); seg2.start = 120.0; seg2.text = "World"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

    with patch("sys.argv", ["convert.py", str(fake_mp4)]):
        with patch("convert.load_model", return_value=mock_model):
            main()

    lines = (tmp_path / "video.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[00:00:03] Hello"
    assert lines[1] == "[00:02:00] World"
