import io
import json
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import asr_server


class TranscriptionPromptTest(unittest.TestCase):
    def test_normalize_prompt_treats_blank_as_absent(self):
        self.assertIsNone(asr_server._normalize_transcription_prompt(None))
        self.assertIsNone(asr_server._normalize_transcription_prompt("   \n\t  "))

    def test_normalize_prompt_strips_and_caps(self):
        long_prompt = "  " + ("x" * (asr_server.MAX_TRANSCRIPTION_PROMPT_CHARS + 20)) + "  "
        normalized = asr_server._normalize_transcription_prompt(long_prompt)

        self.assertEqual(len(normalized), asr_server.MAX_TRANSCRIPTION_PROMPT_CHARS)
        self.assertEqual(normalized, "x" * asr_server.MAX_TRANSCRIPTION_PROMPT_CHARS)

    def test_subprocess_receives_prompt_via_env_not_argv(self):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs["env"]
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"segments": ["ok"], "timings": {}}),
                stderr="",
            )

        with patch.object(asr_server.subprocess, "run", side_effect=fake_run):
            segments, _ = asr_server._run_transcription_subprocess(
                "/tmp/audio.wav",
                "es",
                0.0,
                "ExampleSecretTerm",
            )

        self.assertEqual(segments, ["ok"])
        self.assertNotIn("ExampleSecretTerm", captured["cmd"])
        self.assertEqual(captured["env"]["ASR_TRANSCRIBE_ONCE_PROMPT"], "ExampleSecretTerm")

    def test_persistent_worker_request_includes_prompt(self):
        class FakeStdin:
            def __init__(self):
                self.value = ""

            def write(self, value):
                self.value += value

            def flush(self):
                pass

        fake_stdin = FakeStdin()
        fake_process = SimpleNamespace(stdin=fake_stdin, poll=lambda: None)

        def fake_start():
            asr_server._whisper_worker_process = fake_process

        with (
            patch.object(asr_server, "_start_whisper_worker_locked", side_effect=fake_start),
            patch.object(
                asr_server,
                "_read_whisper_worker_response",
                return_value={"ok": True, "segments": ["ok"], "timings": {}},
            ),
        ):
            segments, _ = asr_server._run_transcription_persistent(
                "/tmp/audio.wav",
                "es",
                0.0,
                "ExampleApp",
            )

        request = json.loads(fake_stdin.value)
        self.assertEqual(segments, ["ok"])
        self.assertEqual(request["prompt"], "ExampleApp")

    def test_transcribe_file_once_passes_initial_prompt(self):
        calls = {}

        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, *args, **kwargs):
                calls.update(kwargs)
                return [SimpleNamespace(text="ok")], None

        with patch("faster_whisper.WhisperModel", FakeModel):
            self.assertEqual(
                asr_server._transcribe_file_once("/tmp/audio.wav", "es", 0.0, "ExampleTerm"),
                ["ok"],
            )

        self.assertEqual(calls["initial_prompt"], "ExampleTerm")

    def test_worker_loop_passes_initial_prompt(self):
        calls = {}

        class FakeModel:
            def transcribe(self, *args, **kwargs):
                calls.update(kwargs)
                return [SimpleNamespace(text="ok")], None

        request = {
            "request_id": "req1",
            "audio_path": "/tmp/audio.wav",
            "language": "es",
            "temperature": 0.0,
            "prompt": "Qwen",
        }
        stdout = io.StringIO()
        with (
            patch.object(asr_server, "load_model", return_value=FakeModel()),
            patch.object(sys, "stdin", io.StringIO(json.dumps(request) + "\n")),
            patch.object(sys, "stdout", stdout),
        ):
            asr_server._transcribe_worker_loop()

        self.assertEqual(calls["initial_prompt"], "Qwen")
        self.assertIn('"ok": true', stdout.getvalue())

    def test_endpoint_accepts_openai_prompt_and_logs_only_count(self):
        secret_prompt = "ExampleSecretTerm"
        captured = {}

        def fake_run_transcription(audio_path, language, temperature, prompt):
            captured["prompt"] = prompt
            return ["raw text"], {}

        with (
            patch.object(asr_server, "ASR_BEARER_TOKEN", "test-token"),
            patch.object(asr_server, "ASR_STOP_LLM_FOR_TRANSCRIPTION", False),
            patch.object(asr_server, "_run_transcription", side_effect=fake_run_transcription),
            patch.object(asr_server, "_postprocess_transcript", side_effect=lambda text, *args: text),
            patch("builtins.print") as print_mock,
        ):
            client = TestClient(asr_server.app)
            response = client.post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer test-token"},
                data={"model": "whisper-1", "language": "es", "prompt": secret_prompt},
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["prompt"], secret_prompt)
        logs = "\n".join(" ".join(str(part) for part in call.args) for call in print_mock.call_args_list)
        self.assertIn(f"prompt_chars={len(secret_prompt)}", logs)
        self.assertNotIn(secret_prompt, logs)


if __name__ == "__main__":
    unittest.main()
