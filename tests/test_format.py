import json
import pytest
import urllib.error
from unittest.mock import MagicMock, patch, mock_open

from format import call_llm


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
