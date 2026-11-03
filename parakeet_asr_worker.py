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
import unicodedata
import wave


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


class LanguageRecoveryError(RuntimeError):
    pass


def _contains_cyrillic(text: str) -> bool:
    return any("CYRILLIC" in unicodedata.name(char, "") for char in text)


def _append_silence(source_path: str, target_path: str):
    with wave.open(source_path, "rb") as source, wave.open(target_path, "wb") as target:
        target.setparams(source.getparams())
        while frames := source.readframes(65536):
            target.writeframesraw(frames)
        target.writeframes(b"\0" * (source.getframerate() // 2 * source.getsampwidth() * source.getnchannels()))


def transcribe_request(model, request: dict, *, diagnostics: dict | None = None) -> list[str]:
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
        if (request.get("language") or "").lower() == "es" and _contains_cyrillic(text):
            started = time.monotonic()
            initial_text = text
            # Preserve speech samples; trailing silence changes short-clip boundary context.
            padded_path = str(Path(directory) / "padded.wav")
            try:
                _append_silence(audio_path, padded_path)
                text = model.transcribe(audio=padded_path, timestamps="none")["text"]
                if not isinstance(text, str) or not text.strip() or _contains_cyrillic(text):
                    raise LanguageRecoveryError()
            except Exception:
                raise LanguageRecoveryError("Spanish transcription recovery failed") from None
            if diagnostics is not None:
                diagnostics.update(initial_raw_text=initial_text, parakeet_language_fallback=1,
                                   language_fallback_ms=round((time.monotonic() - started) * 1000))
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
                    diagnostics = {}
                    segments = transcribe_request(model, request, diagnostics=diagnostics)
                response = {"request_id": request_id, "ok": True, "segments": segments,
                            "timings": {"parakeet_transcribe_ms": round((time.monotonic() - started) * 1000)}}
                if "initial_raw_text" in diagnostics:
                    response["initial_raw_text"] = diagnostics.pop("initial_raw_text")
                    response["timings"].update(diagnostics)
            except LanguageRecoveryError:
                response = {"request_id": request_id, "ok": False,
                            "error_code": "language_recovery_failed"}
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
