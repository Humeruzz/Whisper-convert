import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
LANGUAGE = os.getenv("WHISPER_LANGUAGE") or None


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_transcript(segments: list[tuple[float, str]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for start, text in segments:
            f.write(f"[{format_timestamp(start)}] {text}\n")


def transcribe(model: WhisperModel, input_path: str) -> list[tuple[float, str]]:
    segments, _ = model.transcribe(
        input_path,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return [(seg.start, seg.text.strip()) for seg in segments]
