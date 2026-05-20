import argparse
import os
from pathlib import Path

import whisper
from dotenv import load_dotenv

load_dotenv()

MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
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


def transcribe(model, input_path: str) -> list[tuple[float, str]]:
    result = model.transcribe(
        input_path,
        language=LANGUAGE,
        beam_size=5,
        fp16=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    return [(seg["start"], seg["text"].strip()) for seg in result["segments"]]


def load_model():
    print(f"Loading Whisper model ({MODEL_SIZE})...", end="", flush=True)
    model = whisper.load_model(MODEL_SIZE, device="cuda")
    print(" done")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an MP4 file to a timestamped text transcript using Whisper"
    )
    parser.add_argument("input", help="Path to input MP4 file")
    parser.add_argument(
        "-o", "--output",
        help="Output .txt path (default: same name and location as input)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: '{args.input}' not found")
        return 1

    output_path = args.output or str(Path(args.input).with_suffix(".txt"))

    model = load_model()
    print(f"Transcribing {args.input}...")
    segments = transcribe(model, args.input)
    write_transcript(segments, output_path)
    print(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
