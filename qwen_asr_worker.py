#!/usr/bin/env python3
"""Resident Qwen3-ASR worker. JSON lines on stdin/stdout; model logs use stderr."""

import json
import os
import sys
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
            results = model.transcribe(
                audio=request["audio_path"],
                language=qwen_language(request.get("language")),
                context=request.get("prompt") or "",
            )
            print(json.dumps({
                "request_id": request_id,
                "ok": True,
                "segments": [result.text for result in results],
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
