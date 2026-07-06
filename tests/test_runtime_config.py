import json
import unittest
from unittest.mock import patch

import asr_server


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RuntimeConfigTest(unittest.TestCase):
    def _chat_payload(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Clean text.",
                    },
                }
            ],
            "usage": {},
        }

    def test_cpu_no_postprocess_returns_raw_text(self):
        with (
            patch.object(asr_server, "ASR_DEVICE", "cpu"),
            patch.object(asr_server, "ASR_COMPUTE_TYPE", "int8"),
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", False),
        ):
            self.assertEqual(asr_server._postprocess_transcript("raw text"), "raw text")
            self.assertEqual(asr_server._validate_runtime_config(), [])

    def test_external_postprocess_does_not_start_local_llama(self):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_PROVIDER", "openai-compatible"),
            patch.object(asr_server, "_ensure_llm_server", side_effect=AssertionError("local start")),
            patch.object(asr_server.urllib.request, "urlopen", return_value=FakeResponse(self._chat_payload())),
        ):
            result = asr_server._llm_chat(
                [{"role": "user", "content": "raw text"}],
                max_tokens=64,
                postprocess_mode=True,
            )

        self.assertEqual(result["choices"][0]["message"]["content"], "Clean text.")

    def test_local_llama_provider_starts_local_server(self):
        called = {"ensure": False}

        def fake_ensure():
            called["ensure"] = True

        with (
            patch.object(asr_server, "ASR_POSTPROCESS_PROVIDER", "local-llama"),
            patch.object(asr_server, "_ensure_llm_server", side_effect=fake_ensure),
            patch.object(asr_server.urllib.request, "urlopen", return_value=FakeResponse(self._chat_payload())),
        ):
            asr_server._llm_chat(
                [{"role": "user", "content": "raw text"}],
                max_tokens=64,
                postprocess_mode=True,
            )

        self.assertTrue(called["ensure"])

    def test_gpu_watcher_disabled_by_default_path(self):
        with patch.object(asr_server, "ASR_GPU_WATCH_ENABLED", False):
            self.assertFalse(asr_server._gpu_watch_enabled())

    def test_v16_gpu_watcher_matches_executable_not_arguments(self):
        output = "\n".join(
            [
                "100 /usr/bin/media-server --ffmpeg=/usr/lib/media-ffmpeg/ffmpeg",
                "101 /usr/lib/media-ffmpeg/ffmpeg -hwaccel cuda input.mkv",
            ]
        )
        with (
            patch.object(asr_server, "ASR_GPU_WATCH_PROCESS_PATTERNS", ["ffmpeg"]),
            patch.object(asr_server.os, "getpid", return_value=999),
            patch.object(asr_server.subprocess, "check_output", return_value=output),
        ):
            self.assertEqual(
                asr_server._watched_gpu_processes(),
                [(101, "/usr/lib/media-ffmpeg/ffmpeg -hwaccel cuda input.mkv")],
            )

    def test_cpu_float16_config_warns(self):
        with (
            patch.object(asr_server, "ASR_DEVICE", "cpu"),
            patch.object(asr_server, "ASR_COMPUTE_TYPE", "float16"),
        ):
            self.assertIn("ASR_COMPUTE_TYPE=float16", "\n".join(asr_server._validate_runtime_config()))


if __name__ == "__main__":
    unittest.main()
