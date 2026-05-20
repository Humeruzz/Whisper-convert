import json
import pytest
import urllib.error
from unittest.mock import MagicMock, patch, mock_open

from format import call_llm, format_file


def test_call_llm_returns_cleaned_text():
    response_body = json.dumps({
        "choices": [{"message": {"content": "Cleaned text."}}]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = call_llm("raw text", mode="format")

    assert result == "Cleaned text."


def test_call_llm_summarize_uses_correct_temperature():
    response_body = json.dumps({
        "choices": [{"message": {"content": "Summary."}}]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open_url:
        call_llm("raw text", mode="summarize")

    req = mock_open_url.call_args[0][0]
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["temperature"] == 0.3


def test_call_llm_format_uses_correct_temperature():
    response_body = json.dumps({
        "choices": [{"message": {"content": "Formatted."}}]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open_url:
        call_llm("raw text", mode="format")

    req = mock_open_url.call_args[0][0]
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["temperature"] == 0.1


def test_call_llm_url_error_raises():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(urllib.error.URLError):
            call_llm("text", mode="format")


def test_call_llm_timeout_raises():
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        with pytest.raises(TimeoutError):
            call_llm("text", mode="format")


def test_call_llm_wraps_text_in_transcription_tags():
    response_body = json.dumps({
        "choices": [{"message": {"content": "ok"}}]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open_url:
        call_llm("my text", mode="format")

    req = mock_open_url.call_args[0][0]
    payload = json.loads(req.data.decode("utf-8"))
    user_msg = payload["messages"][1]["content"]
    assert "<transcription>" in user_msg
    assert "my text" in user_msg


def test_format_file_reads_input_and_writes_output(tmp_path):
    input_file = tmp_path / "in.txt"
    input_file.write_text("[00:00:01] Hello um world.\n[00:00:05] Goodbye.\n", encoding="utf-8")
    output_file = tmp_path / "out.txt"

    with patch("format.call_llm", return_value="Hello world.\nGoodbye."):
        format_file(str(input_file), str(output_file), mode="format")

    assert output_file.read_text(encoding="utf-8") == "Hello world.\nGoodbye."


def test_format_file_passes_full_content_to_llm(tmp_path):
    input_file = tmp_path / "in.txt"
    content = "[00:00:01] Some text.\n[00:00:05] More text.\n"
    input_file.write_text(content, encoding="utf-8")

    with patch("format.call_llm", return_value="cleaned") as mock_llm:
        format_file(str(input_file), str(tmp_path / "out.txt"), mode="format")

    mock_llm.assert_called_once_with(content, mode="format")


def test_format_file_llm_error_prints_and_exits(tmp_path, capsys):
    input_file = tmp_path / "in.txt"
    input_file.write_text("text", encoding="utf-8")

    with patch("format.call_llm", side_effect=urllib.error.URLError("connection refused")):
        result = format_file(str(input_file), str(tmp_path / "out.txt"), mode="format")

    assert result == 1
    assert "unavailable" in capsys.readouterr().out.lower()
