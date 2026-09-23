import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import asr_server
import qwen_asr_worker


class QwenRoutingTest(unittest.TestCase):
    def test_audio_model_catalog_lists_only_enabled_transcription_models(self):
        with (
            patch.object(asr_server, "ASR_BEARER_TOKEN", "test-token"),
            patch.object(asr_server, "ASR_QWEN_ENABLED", True),
            patch.object(asr_server, "ASR_QWEN_MODEL_ID", "future-qwen-id"),
            patch.object(asr_server, "ASR_POSTPROCESS_MODEL", "cleanup-only"),
        ):
            response = TestClient(asr_server.app).get(
                "/v1/audio/models", headers={"Authorization": "Bearer test-token"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        self.assertEqual(
            [model["id"] for model in response.json()["data"]],
            ["whisper-1", "future-qwen-id"],
        )

    def test_audio_model_catalog_is_authenticated_and_hides_disabled_qwen(self):
        with (
            patch.object(asr_server, "ASR_BEARER_TOKEN", "test-token"),
            patch.object(asr_server, "ASR_QWEN_ENABLED", False),
        ):
            client = TestClient(asr_server.app)
            self.assertEqual(client.get("/v1/audio/models").status_code, 401)
            response = client.get(
                "/v1/audio/models", headers={"Authorization": "Bearer test-token"}
            )
        self.assertEqual([model["id"] for model in response.json()["data"]], ["whisper-1"])

    def test_qwen_language_converts_openai_codes(self):
        self.assertEqual(qwen_asr_worker.qwen_language("en"), "English")
        self.assertEqual(qwen_asr_worker.qwen_language("es"), "Spanish")
        self.assertIsNone(qwen_asr_worker.qwen_language("auto"))

    def test_legacy_model_names_still_use_whisper(self):
        with patch.object(asr_server, "ASR_QWEN_ENABLED", True):
            self.assertEqual(asr_server._transcription_engine("whisper-1"), "whisper")
            self.assertEqual(asr_server._transcription_engine("large-v3"), "whisper")
            self.assertEqual(asr_server._transcription_engine(None), "whisper")
            self.assertEqual(asr_server._transcription_engine("qwen3-asr-1.7b"), "qwen")

    def test_qwen_disabled_rejects_selector(self):
        with patch.object(asr_server, "ASR_QWEN_ENABLED", False):
            with self.assertRaises(asr_server.HTTPException) as raised:
                asr_server._transcription_engine("qwen3-asr-1.7b")
        self.assertEqual(raised.exception.status_code, 400)

    def test_qwen_worker_ignores_whisper_library_path(self):
        fake_process = SimpleNamespace(pid=123, poll=lambda: None)
        with (
            patch.dict(os.environ, {"LD_LIBRARY_PATH": "/whisper/cudnn"}),
            patch.object(asr_server, "ASR_QWEN_CUDA_VISIBLE_DEVICES", "qwen-gpu"),
            patch.object(asr_server.subprocess, "Popen", return_value=fake_process) as popen,
        ):
            asr_server._qwen_worker_process = None
            try:
                asr_server._start_qwen_worker_locked()
            finally:
                asr_server._qwen_worker_process = None
        worker_env = popen.call_args.kwargs["env"]
        self.assertNotIn("LD_LIBRARY_PATH", worker_env)
        self.assertEqual(worker_env["CUDA_VISIBLE_DEVICES"], "qwen-gpu")

    def test_endpoint_routes_qwen_and_preserves_context(self):
        received = {}

        def fake_qwen(audio_path, language, temperature, prompt):
            received.update(language=language, temperature=temperature, prompt=prompt)
            return ["Hello, world."], {"qwen_transcribe_ms": 12}

        with (
            patch.object(asr_server, "ASR_BEARER_TOKEN", "test-token"),
            patch.object(asr_server, "ASR_QWEN_ENABLED", True),
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", False),
            patch.object(asr_server, "_run_qwen_transcription", side_effect=fake_qwen),
            patch.object(asr_server, "_run_transcription", side_effect=AssertionError("Whisper called")),
        ):
            response = TestClient(asr_server.app).post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer test-token"},
                data={"model": "qwen3-asr-1.7b", "language": "en", "prompt": "Project term"},
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Hello, world.")
        self.assertEqual(received, {"language": "en", "temperature": 0.0, "prompt": "Project term"})
        self.assertEqual(response.json()["timings"]["qwen_transcribe_ms"], 12)

    def test_short_translation_is_rejected(self):
        result = {"choices": [{"message": {"content": "Hola mundo."}}], "usage": {}}
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", return_value=result),
        ):
            self.assertEqual(asr_server._postprocess_transcript("Hello world"), "Hello world")

    def test_short_punctuation_only_cleanup_survives(self):
        result = {"choices": [{"message": {"content": "Hello, world."}}], "usage": {}}
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", return_value=result),
        ):
            self.assertEqual(asr_server._postprocess_transcript("hello world"), "Hello, world.")


if __name__ == "__main__":
    unittest.main()
