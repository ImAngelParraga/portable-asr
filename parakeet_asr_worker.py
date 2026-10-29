#!/usr/bin/env python3
"""Isolated Photon transcription worker; JSON lines on stdin/stdout."""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def transcribe_request(model, request: dict) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="parakeet-audio-") as directory:
        audio_path = str(Path(directory) / "audio.wav")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", request["audio_path"], "-map", "0:a:0", "-vn", "-ac", "1",
             "-ar", "16000", "-c:a", "pcm_s16le", audio_path],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, check=True,
        )
        result = model.transcribe(audio=audio_path, timestamps="none")
        text = result["text"]
        if not isinstance(text, str):
            raise ValueError("Invalid transcription text")
        return [text]


def main():
    model = None
    print(json.dumps({"ready": True}), flush=True)
    try:
        for line in sys.stdin:
            request_id = None
            try:
                request = json.loads(line)
                request_id = request["request_id"]
                started = time.monotonic()
                # Library output must not mix with the response protocol.
                with redirect_stdout(sys.stderr):
                    if model is None:
                        import moondream as md
                        model = md.photon(
                            os.environ.get("ASR_PARAKEET_CHECKPOINT", "moondream/parakeet-ultra"),
                            device="cuda",
                        )
                    segments = transcribe_request(model, request)
                response = {"request_id": request_id, "ok": True, "segments": segments,
                            "timings": {"parakeet_transcribe_ms": round((time.monotonic() - started) * 1000)}}
            except Exception as exc:
                response = {"request_id": request_id, "ok": False,
                            "error": f"Parakeet transcription failed ({type(exc).__name__})"}
            print(json.dumps(response), flush=True)
    finally:
        if model is not None:
            with redirect_stdout(sys.stderr):
                model.close()


if __name__ == "__main__":
    main()
