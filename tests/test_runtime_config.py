import json
import unittest
from unittest.mock import Mock, mock_open, patch

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

    def test_v46_cleanup_penalty_is_configurable_without_affecting_general_chat(self):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_PROVIDER", "openai-compatible"),
            patch.object(asr_server, "ASR_POSTPROCESS_REPEAT_PENALTY", 1.0),
            patch.object(asr_server.urllib.request, "urlopen", return_value=FakeResponse(self._chat_payload())) as request,
        ):
            asr_server._llm_chat([{"role": "user", "content": "Muy, muy bien."}],
                                 max_tokens=64, postprocess_mode=True)
            cleanup = json.loads(request.call_args.args[0].data)
            asr_server._llm_chat([{"role": "user", "content": "Hello."}], max_tokens=64)
            chat = json.loads(request.call_args.args[0].data)

        self.assertEqual(cleanup["repeat_penalty"], 1.0)
        self.assertEqual(cleanup["presence_penalty"], 0.0)
        self.assertEqual(cleanup["frequency_penalty"], 0.0)
        self.assertTrue(cleanup["cache_prompt"])
        self.assertFalse(cleanup["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(chat["repeat_penalty"], 1.18)
        self.assertNotIn("presence_penalty", chat)
        self.assertNotIn("frequency_penalty", chat)

    def test_v47_copy_speculation_is_opt_in_on_local_server_start(self):
        for spec_type in ("none", "ngram-mod"):
            with (
                self.subTest(spec_type=spec_type),
                patch.object(asr_server, "ASR_LLM_SPEC_TYPE", spec_type),
                patch.object(asr_server, "_llm_process", None),
                patch.object(asr_server, "_llm_health", side_effect=[False, True]),
                patch.object(asr_server, "_gpu_watch_enabled", return_value=False),
                patch.object(asr_server, "open", mock_open(), create=True),
                patch.object(asr_server.Path, "write_text"),
                patch.object(asr_server.subprocess, "Popen", return_value=Mock(poll=Mock(return_value=None))) as spawn,
            ):
                asr_server._start_llm_server()
                command = spawn.call_args.args[0]
                self.assertEqual(command[command.index("-hf") + 1], asr_server.ASR_POSTPROCESS_MODEL)
                self.assertNotIn("--model-draft", command)
                if spec_type == "none":
                    self.assertNotIn("--spec-type", command)
                else:
                    self.assertEqual(command.count("--spec-type"), 1)
                    self.assertEqual(command[command.index("--spec-type") + 1], "ngram-mod")

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
