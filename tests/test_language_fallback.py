import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from fastapi.testclient import TestClient
import asr_server as server
import parakeet_asr_worker as worker


class LanguageFallbackTest(unittest.TestCase):
    def decode(self, command, **kwargs):
        with wave.open(command[-1], "wb") as out:
            out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            out.writeframes(b"\x01\x02" * 1600)

    def test_v53_retries_same_model_with_original_samples_and_half_second_silence(self):
        paths = []
        original_samples = b"\x01\x02" * 1600

        def recognize(**kwargs):
            self.assertEqual(set(kwargs), {"audio", "timestamps"})
            path = Path(kwargs["audio"])
            paths.append(path)
            with wave.open(str(path), "rb") as audio:
                samples = audio.readframes(audio.getnframes())
                self.assertEqual(samples, original_samples if len(paths) == 1
                                 else original_samples + b"\0" * 16000)
            return {"text": "Донор." if len(paths) == 1 else "dolor"}

        model = Mock()
        model.transcribe.side_effect = recognize
        diagnostics = {}
        with patch.object(worker.subprocess, "run", side_effect=self.decode):
            self.assertEqual(worker.transcribe_request(model, {"audio_path": "/clip.m4a", "language": "es"},
                                                      diagnostics=diagnostics), ["dolor"])
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertEqual(diagnostics["initial_raw_text"], "Донор.")
        self.assertEqual(diagnostics["parakeet_language_fallback"], 1)
        self.assertTrue(all(not p.exists() for p in paths))
        model.close.assert_not_called()

    def test_v53_ordinary_spanish_and_other_hints_skip_retry(self):
        for language, text in (("es", "Dolor."), (None, "Донор."), ("ru", "Донор."),
                               ("es", "El código α vale 2."), ("en", "Донор.")):
            with self.subTest(language=language, text=text):
                model = Mock()
                model.transcribe.return_value = {"text": text}
                with patch.object(worker.subprocess, "run", side_effect=self.decode):
                    self.assertEqual(worker.transcribe_request(model, {"audio_path": "/clip", "language": language}), [text])
                self.assertEqual(model.transcribe.call_count, 1)

    def test_v53_empty_wrong_script_invalid_or_failed_retry_fails_closed_and_cleans_audio(self):
        for recovered in ("", "Донор.", None, RuntimeError("private text")):
            with self.subTest(recovered=recovered):
                model = Mock()
                model.transcribe.side_effect = [{"text": "Tengo Донор."},
                                               recovered if isinstance(recovered, Exception) else {"text": recovered}]
                with patch.object(worker.subprocess, "run", side_effect=self.decode):
                    with self.assertRaisesRegex(worker.LanguageRecoveryError, "Spanish transcription recovery failed"):
                        worker.transcribe_request(model, {"audio_path": "/clip", "language": "ES"})
                self.assertEqual(model.transcribe.call_count, 2)
                for call in model.transcribe.call_args_list:
                    self.assertFalse(Path(call.kwargs["audio"]).exists())

    def test_v53_worker_failure_is_sanitized_and_next_request_keeps_model(self):
        model = Mock()
        requests = "".join(json.dumps({"request_id": str(i), "audio_path": "/clip", "language": "es"}) + "\n"
                           for i in range(2))
        with (
            patch.object(worker, "load_model", return_value=model) as load,
            patch.object(worker.sys, "stdin", io.StringIO(requests)),
            patch.object(worker, "transcribe_request", side_effect=[worker.LanguageRecoveryError("private text"), ["Hola"]]),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            worker.main()
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(responses[1]["error_code"], "language_recovery_failed")
        self.assertEqual(responses[2]["segments"], ["Hola"])
        self.assertNotIn("private", output.getvalue())
        load.assert_called_once()

    def test_v53_worker_protocol_forwards_language_and_retains_failure_residency(self):
        process = SimpleNamespace(pid=123, stdin=io.StringIO(), poll=lambda: None)
        with (
            patch.multiple(server, _parakeet_worker_process=process, ASR_PARAKEET_KEEP_WARM=True),
            patch.object(server, "_start_parakeet_worker_locked"),
            patch.object(server, "_stop_parakeet_worker") as stop,
            patch.object(server, "_read_whisper_worker_response", return_value={
                "ok": False, "error_code": "language_recovery_failed"}),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as error:
                server._run_parakeet_transcription("/audio", "es", 0)
            self.assertEqual(error.exception.status_code, 503)
            stop.assert_not_called()
        self.assertEqual(json.loads(process.stdin.getvalue())["language"], "es")

    def test_v53_endpoint_archives_both_recognitions_without_switching_models(self):
        events = []
        def recognize(*args):
            events.append("parakeet")
            model = Mock()
            model.transcribe.side_effect = [{"text": "Донор."}, {"text": "dolor"}]
            diagnostics = {}
            with patch.object(worker.subprocess, "run", side_effect=self.decode):
                segments = worker.transcribe_request(model, {"audio_path": args[0], "language": args[1]}, diagnostics=diagnostics)
            return segments, {"_language_fallback_raw_text": diagnostics.pop("initial_raw_text"), **diagnostics}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.multiple(server, ASR_BEARER_TOKEN="test-token", ASR_PARAKEET_ENABLED=True,
                           ASR_SINGLE_RESIDENT_MODEL=True, ASR_AUDIO_HISTORY_ENABLED=True,
                           ASR_AUDIO_HISTORY_DIR=directory, ASR_STOP_LLM_FOR_TRANSCRIPTION=False,
                           ASR_PARAKEET_STOP_LLM_FOR_TRANSCRIPTION=False),
            patch.object(server, "_stop_whisper_worker"), patch.object(server, "_stop_qwen_worker"),
            patch.object(server, "unload_model"),
            patch.object(server, "_stop_parakeet_worker") as stop,
            patch.object(server, "_run_parakeet_transcription", side_effect=recognize),
            patch.object(server, "_run_transcription") as whisper,
            patch.object(server, "_postprocess_transcript", return_value="Dolor.") as cleanup,
            contextlib.redirect_stdout(io.StringIO()) as logs,
        ):
            response = TestClient(server.app).post(
                "/v1/audio/transcriptions", headers={"Authorization": "Bearer test-token"},
                files={"file": ("clip.m4a", b"audio", "audio/mp4")},
                data={"model": "parakeet-ultra", "language": "es"},
            )
            record = json.loads(next(Path(directory).glob("*/metadata.json")).read_text())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Dolor.")
        self.assertNotIn("_language_fallback_raw_text", response.json()["timings"])
        self.assertEqual(cleanup.call_args.args[0], "dolor")
        self.assertEqual(record["initial_raw_text"], "Донор.")
        self.assertEqual(record["raw_text"], "dolor")
        self.assertEqual(record["recognition_engine"], "parakeet")
        self.assertEqual(record["fallback_reason"], "spanish_cyrillic")
        self.assertNotIn("Донор", logs.getvalue())
        self.assertNotIn("dolor", logs.getvalue())
        whisper.assert_not_called()
        stop.assert_not_called()
