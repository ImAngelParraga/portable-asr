import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import asr_server
import parakeet_asr_worker as worker


class ParakeetRoutingTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            asr_server, ASR_BEARER_TOKEN="test-token", ASR_PARAKEET_ENABLED=True,
            ASR_PARAKEET_MODEL_ID="parakeet-ultra", ASR_STOP_LLM_FOR_TRANSCRIPTION=False,
            ASR_PARAKEET_STOP_LLM_FOR_TRANSCRIPTION=False, ASR_AUDIO_HISTORY_ENABLED=False,
        ))
        self.client = TestClient(asr_server.app)
        self.headers = {"Authorization": "Bearer test-token"}

    def upload(self, model="parakeet-ultra"):
        return self.client.post(
            "/v1/audio/transcriptions", headers=self.headers,
            files={"file": ("recording.m4a", b"audio", "audio/mp4")},
            data={"model": model, "language": "es", "prompt": "private vocabulary"},
        )

    def test_v38_catalog_authentication_and_optional_visibility(self):
        with patch.object(asr_server, "_start_parakeet_worker_locked") as start:
            self.assertEqual(self.client.get("/v1/audio/models").status_code, 401)
            for enabled in (False, True):
                with patch.object(asr_server, "ASR_PARAKEET_ENABLED", enabled):
                    for path in ("/v1/audio/models", "/v1/models"):
                        ids = [m["id"] for m in self.client.get(path, headers=self.headers).json()["data"]]
                        self.assertEqual("parakeet-ultra" in ids, enabled)
                        self.assertEqual(ids[0], "whisper-1")
            start.assert_not_called()

    def test_v38_disabled_selector_rejects_without_loading(self):
        with (patch.object(asr_server, "ASR_PARAKEET_ENABLED", False),
              patch.object(asr_server, "_run_transcription") as whisper):
            response = self.upload()
        self.assertEqual(response.status_code, 400)
        whisper.assert_not_called()

    def test_v38_default_qwen_and_custom_selector_routing(self):
        with patch.object(asr_server, "ASR_QWEN_ENABLED", True):
            for legacy in (None, "whisper-1", "large-v3-turbo", "legacy-name"):
                self.assertEqual(asr_server._transcription_engine(legacy), "whisper")
            self.assertEqual(asr_server._transcription_engine(asr_server.ASR_QWEN_MODEL_ID), "qwen")
        with patch.object(asr_server, "ASR_PARAKEET_MODEL_ID", "custom-parakeet"):
            self.assertEqual(asr_server._transcription_engine("custom-parakeet"), "parakeet")

    def test_v38_endpoint_preserves_cleanup_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            with (
                patch.multiple(asr_server, ASR_AUDIO_HISTORY_ENABLED=True,
                               ASR_AUDIO_HISTORY_DIR=directory,
                               ASR_PARAKEET_STOP_LLM_FOR_TRANSCRIPTION=True,
                               ASR_POSTPROCESS_PROVIDER="local-llama"),
                patch.object(asr_server, "_stop_llm_server", side_effect=lambda *a: events.append("stop-cleanup")),
                patch.object(asr_server, "_run_parakeet_transcription", side_effect=lambda *a: (events.append("parakeet") or ["raw Spanish"], {"parakeet_transcribe_ms": 35})),
                patch.object(asr_server, "_postprocess_transcript", side_effect=lambda *a: events.append("cleanup") or "Final Spanish."),
                patch.object(asr_server, "_run_transcription") as whisper,
                patch.object(asr_server, "_run_qwen_transcription") as qwen,
            ):
                response = self.upload()
                record = json.loads(next(Path(directory).glob("*/metadata.json")).read_text())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], "Final Spanish.")
            self.assertEqual(response.json()["timings"]["parakeet_transcribe_ms"], 35)
            self.assertEqual(events, ["stop-cleanup", "parakeet", "cleanup"])
            self.assertEqual(record["engine"], "parakeet")
            self.assertEqual(record["asr_model"], asr_server.ASR_PARAKEET_CHECKPOINT)
            self.assertEqual(record["raw_text"], "raw Spanish")
            self.assertEqual(record["final_text"], "Final Spanish.")
            whisper.assert_not_called()
            qwen.assert_not_called()

    def test_v40_shared_gpu_setting_does_not_stop_remote_cleanup(self):
        with (
            patch.multiple(asr_server, ASR_PARAKEET_STOP_LLM_FOR_TRANSCRIPTION=True,
                           ASR_POSTPROCESS_PROVIDER="openai-compatible"),
            patch.object(asr_server, "_stop_llm_server") as stop,
            patch.object(asr_server, "_run_parakeet_transcription", return_value=(["Hola"], {})),
            patch.object(asr_server, "_postprocess_transcript", return_value="Hola"),
        ):
            self.assertEqual(self.upload().status_code, 200)
            stop.assert_not_called()

    def test_v40_worker_has_isolated_libraries_and_explicit_gpu_cache(self):
        process = SimpleNamespace(pid=123, poll=lambda: None)
        with (
            patch.dict(os.environ, {"LD_LIBRARY_PATH": "/whisper/cudnn", "HUGGINGFACE_HUB_CACHE": "/other/cache"}),
            patch.multiple(asr_server, _parakeet_worker_process=None,
                           ASR_PARAKEET_CUDA_VISIBLE_DEVICES="explicit-gpu",
                           ASR_PARAKEET_CACHE_DIR="/cache/parakeet"),
            patch.object(asr_server.subprocess, "Popen", return_value=process) as popen,
        ):
            asr_server._start_parakeet_worker_locked()
            env = popen.call_args.kwargs["env"]
            self.assertNotIn("LD_LIBRARY_PATH", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "explicit-gpu")
            self.assertEqual(env["HF_HOME"], "/cache/parakeet")
            self.assertEqual(env["HF_HUB_CACHE"], "/cache/parakeet/hub")
            self.assertEqual(env["HUGGINGFACE_HUB_CACHE"], "/cache/parakeet/hub")
            self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.DEVNULL)

    def test_v40_nonresident_worker_releases_memory_before_return(self):
        process = SimpleNamespace(pid=123, stdin=io.StringIO(), poll=lambda: None)
        for keep_warm in (False, True):
            with (
                self.subTest(keep_warm=keep_warm),
                patch.multiple(asr_server, _parakeet_worker_process=process,
                               ASR_PARAKEET_KEEP_WARM=keep_warm),
                patch.object(asr_server, "_start_parakeet_worker_locked"),
                patch.object(asr_server, "_stop_parakeet_worker") as stop,
                patch.object(asr_server, "_read_whisper_worker_response", return_value={
                    "ok": True, "segments": ["Hola"], "timings": {"parakeet_transcribe_ms": 35}}),
            ):
                segments, _ = asr_server._run_parakeet_transcription("/audio.wav", "es", .7, "private term")
                self.assertEqual(segments, ["Hola"])
                self.assertEqual(stop.called, not keep_warm)
        for line in process.stdin.getvalue().splitlines():
            self.assertEqual(set(json.loads(line)), {"request_id", "audio_path"})

    def test_v40_failure_stops_resident_worker_without_private_errors(self):
        process = SimpleNamespace(pid=123, stdin=io.StringIO(), poll=lambda: None)
        with (
            patch.multiple(asr_server, _parakeet_worker_process=process, ASR_PARAKEET_KEEP_WARM=True),
            patch.object(asr_server, "_start_parakeet_worker_locked"),
            patch.object(asr_server, "_stop_parakeet_worker") as stop,
            patch.object(asr_server, "_read_whisper_worker_response", return_value={
                "ok": False, "error": "private audio text"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "^Parakeet transcription failed$"):
                asr_server._run_parakeet_transcription("/audio.wav", None, 0)
            stop.assert_called_once()

    def test_v40_owned_worker_does_not_block_cleanup(self):
        with (
            patch.object(asr_server, "_parakeet_worker_process", SimpleNamespace(pid=123, poll=lambda: None)),
            patch.object(asr_server, "_llm_server_processes", return_value=set()),
            patch.object(asr_server, "_gpu_compute_apps", return_value=[(123, "python", "2193 MiB"), (456, "external", "1 MiB")]),
        ):
            self.assertEqual(asr_server._external_gpu_users(), [(456, "external", "1 MiB")])

    def test_v40_worker_stop_waits_for_cuda_process_exit(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 10), None]
        with patch.object(asr_server, "_parakeet_worker_process", process):
            asr_server._stop_parakeet_worker()
            self.assertIsNone(asr_server._parakeet_worker_process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        process.stdin.close.assert_called_once()
        process.stdout.close.assert_called_once()

    def test_v41_exited_cleanup_pid_does_not_delay_gpu_handoff(self):
        with (
            patch.object(asr_server.os, "kill"),
            patch.object(Path, "read_text", return_value="123 (llama-server) Z 1 2 3"),
        ):
            self.assertFalse(asr_server._process_exists(123))
        with (
            patch.object(asr_server.os, "kill"),
            patch.object(Path, "read_text", return_value="123 (llama-server) S 1 2 3"),
        ):
            self.assertTrue(asr_server._process_exists(123))

    def test_v41_cleanup_shutdown_reaps_owned_child(self):
        process = Mock()
        process.pid = 123
        process.poll.return_value = None
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.multiple(asr_server, _llm_process=process, ASR_LLM_PID_FILE=str(Path(directory) / "pid")),
            patch.object(asr_server, "_llm_server_processes", side_effect=[{123}, set(), set()]),
            patch.object(asr_server.os, "kill"),
            patch.object(asr_server.time, "sleep") as sleep,
        ):
            asr_server._stop_llm_server()
            self.assertGreaterEqual(process.poll.call_count, 2)
            sleep.assert_not_called()


class ParakeetWorkerTest(unittest.TestCase):
    def test_v39_decodes_wav_and_m4a_and_cleans_temporary_audio(self):
        for extension in ("wav", "m4a"):
            model = Mock()
            captured = []

            def decode(command, **kwargs):
                self.assertIn("16000", command)
                self.assertEqual(command[command.index("-ac") + 1], "1")
                Path(command[-1]).write_bytes(b"pcm")

            def transcribe(**kwargs):
                captured.append(Path(kwargs["audio"]))
                self.assertTrue(captured[-1].exists())
                self.assertEqual(set(kwargs), {"audio", "timestamps"})
                return {"text": "Hola"}

            model.transcribe.side_effect = transcribe
            with patch.object(worker.subprocess, "run", side_effect=decode):
                self.assertEqual(worker.transcribe_request(model, {"audio_path": f"/audio.{extension}"}), ["Hola"])
            self.assertFalse(captured[0].exists())

    def test_v39_decode_failure_never_invokes_model(self):
        model = Mock()
        with patch.object(worker.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            with self.assertRaises(subprocess.CalledProcessError):
                worker.transcribe_request(model, {"audio_path": "/bad.m4a"})
        model.transcribe.assert_not_called()

    def test_v39_inference_failure_also_removes_decoded_audio(self):
        paths = []

        def decode(command, **kwargs):
            paths.append(Path(command[-1]))
            paths[-1].write_bytes(b"pcm")

        model = Mock()
        model.transcribe.side_effect = RuntimeError("inference failed")
        with patch.object(worker.subprocess, "run", side_effect=decode):
            with self.assertRaises(RuntimeError):
                worker.transcribe_request(model, {"audio_path": "/audio.wav"})
        self.assertFalse(paths[0].exists())

    def test_v39_worker_protocol_sanitizes_library_failures(self):
        model = Mock()
        fake_module = SimpleNamespace(photon=Mock(return_value=model))
        input_data = json.dumps({"request_id": "r1", "audio_path": "/private.wav"}) + "\n"
        output = io.StringIO()
        with (
            patch.dict("sys.modules", {"moondream": fake_module}),
            patch.object(worker.sys, "stdin", io.StringIO(input_data)),
            contextlib.redirect_stdout(output),
            patch.object(worker, "transcribe_request", side_effect=RuntimeError("private transcript")),
        ):
            worker.main()
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertFalse(messages[-1]["ok"])
        self.assertEqual(messages[-1]["request_id"], "r1")
        self.assertNotIn("private", output.getvalue())
        model.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
