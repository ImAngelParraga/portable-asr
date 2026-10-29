#!/usr/bin/env python3
"""Isolated Parakeet transcription worker; JSON lines on stdin/stdout."""

from contextlib import redirect_stdout
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


class CppParakeet:
    def __init__(self, library: str, checkpoint: str):
        if not library or not checkpoint:
            raise ValueError("Parakeet C++ library and GGUF paths are required")
        self.library = ctypes.CDLL(library)
        self.library.parakeet_capi_load.argtypes = [ctypes.c_char_p]
        self.library.parakeet_capi_load.restype = ctypes.c_void_p
        self.library.parakeet_capi_transcribe_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.library.parakeet_capi_transcribe_path.restype = ctypes.c_void_p
        self.library.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
        self.library.parakeet_capi_free_string.restype = None
        self.library.parakeet_capi_free.argtypes = [ctypes.c_void_p]
        self.library.parakeet_capi_free.restype = None
        self.context = self.library.parakeet_capi_load(os.fsencode(checkpoint))
        if not self.context:
            raise RuntimeError("Could not load Parakeet GGUF")

    def transcribe(self, *, audio: str, timestamps: str = "none") -> dict:
        result = self.library.parakeet_capi_transcribe_path(self.context, os.fsencode(audio), 0)
        if not result:
            raise RuntimeError("Parakeet C++ transcription failed")
        try:
            return {"text": ctypes.string_at(result).decode("utf-8")}
        finally:
            self.library.parakeet_capi_free_string(result)

    def close(self):
        if self.context:
            self.library.parakeet_capi_free(self.context)
            self.context = None


def load_model():
    backend = os.environ.get("ASR_PARAKEET_BACKEND", "photon")
    if backend == "cpp":
        return CppParakeet(os.environ.get("ASR_PARAKEET_LIBRARY", ""),
                           os.environ.get("ASR_PARAKEET_GGUF", ""))
    if backend != "photon":
        raise ValueError("Unknown Parakeet backend")
    import moondream as md
    return md.photon(os.environ.get("ASR_PARAKEET_CHECKPOINT", "moondream/parakeet-ultra"), device="cuda")


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
                        model = load_model()
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
