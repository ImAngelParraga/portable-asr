#!/usr/bin/env python3
"""Resident Qwen3-ASR worker. JSON lines on stdin/stdout; model logs use stderr."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


LANGUAGE_NAMES = {
    "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian",
    "fil": "Filipino", "fi": "Finnish", "fr": "French", "hi": "Hindi",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sv": "Swedish",
    "th": "Thai", "tr": "Turkish", "vi": "Vietnamese", "yue": "Cantonese",
    "zh": "Chinese",
}


def qwen_language(language: str | None) -> str | None:
    if not language or language.lower() == "auto":
        return None
    return LANGUAGE_NAMES.get(language.lower(), language)


def decode_audio_to_wav(source_path: str, output_path: str) -> None:
    """Decode client audio to the PCM format Qwen's file loader can read."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", source_path, "-map", "0:a:0", "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output_path,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )


def transcribe_request(model, request: dict) -> list[str]:
    """Keep decoded audio alive through inference, then remove it."""
    with tempfile.TemporaryDirectory(prefix="qwen-audio-") as temp_dir:
        audio_path = str(Path(temp_dir) / "audio.wav")
        decode_audio_to_wav(request["audio_path"], audio_path)
        results = model.transcribe(
            audio=audio_path,
            language=qwen_language(request.get("language")),
            context=request.get("prompt") or "",
        )
        return [result.text for result in results]


def main() -> None:
    import torch
    from qwen_asr import Qwen3ASRModel

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-ASR CUDA device unavailable")
    torch.set_num_threads(2)
    model = Qwen3ASRModel.from_pretrained(
        os.environ.get("ASR_QWEN_CHECKPOINT", "Qwen/Qwen3-ASR-1.7B"),
        dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        max_inference_batch_size=1,
        max_new_tokens=512,
    )
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            started = time.monotonic()
            segments = transcribe_request(model, request)
            print(json.dumps({
                "request_id": request_id,
                "ok": True,
                "segments": segments,
                "timings": {"qwen_transcribe_ms": round((time.monotonic() - started) * 1000)},
            }), flush=True)
        except Exception as exc:
            # Model exceptions can contain request data. Keep worker output private.
            print(json.dumps({
                "request_id": request_id,
                "ok": False,
                "error": f"Qwen transcription failed ({type(exc).__name__})",
            }), flush=True)


if __name__ == "__main__":
    main()
