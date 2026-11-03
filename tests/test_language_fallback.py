import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi import HTTPException
import asr_server as server


class LanguageFallbackTest(unittest.TestCase):
    def dispatch(self, language="es", text="Донор.", recovered="Dolor.", single=False):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.multiple(server, ASR_SINGLE_RESIDENT_MODEL=single,
                                           ASR_WHISPER_KEEP_WARM=True))
        for name in ("_stop_whisper_worker", "_stop_qwen_worker", "_stop_parakeet_worker", "unload_model"):
            stack.enter_context(patch.object(server, name))
        parakeet = stack.enter_context(patch.object(server, "_run_parakeet_transcription",
                         return_value=([text], {"parakeet_transcribe_ms": 12})))
        whisper = stack.enter_context(patch.object(server, "_run_transcription",
                         return_value=([recovered], {"whisper_transcribe_ms": 30})))
        return parakeet, whisper

    def test_v53_recognizes_original_audio_in_spanish_before_cleanup(self):
        _, whisper = self.dispatch()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            segments, timings = server._dispatch_transcription("parakeet", "/original.m4a", "es", .8, "hint")
        self.assertEqual(segments, ["Dolor."])
        whisper.assert_called_once_with("/original.m4a", "es", 0.0, "hint")
        self.assertEqual(timings["parakeet_language_fallback"], 1)
        self.assertEqual(timings["parakeet_transcribe_ms"], 12)
        self.assertEqual(timings["whisper_transcribe_ms"], 30)
        self.assertEqual(timings["_language_fallback_raw_text"], "Донор.")
        self.assertNotIn("Донор", output.getvalue())

    def test_v53_ordinary_spanish_auto_and_russian_stay_on_parakeet(self):
        for language, text in (("es", "Dolor."), (None, "Донор."), ("ru", "Донор."),
                               ("es", "El código α vale 2."), ("en", "Донор.")):
            with self.subTest(language=language, text=text):
                _, whisper = self.dispatch(language, text)
                segments, _ = server._dispatch_transcription("parakeet", "/audio", language, 0)
                self.assertEqual(segments, [text])
                whisper.assert_not_called()

    def test_v53_mixed_cyrillic_also_retries(self):
        _, whisper = self.dispatch(text="Tengo Донор.")
        server._dispatch_transcription("parakeet", "/audio", "ES", 0)
        whisper.assert_called_once()

    def test_v53_empty_wrong_script_and_failed_retry_fail_closed(self):
        for recovered in ("", "Донор."):
            with self.subTest(recovered=recovered):
                self.dispatch(recovered=recovered)
                with self.assertRaises(HTTPException) as error:
                    server._dispatch_transcription("parakeet", "/audio", "es", 0)
                self.assertEqual(error.exception.status_code, 503)
        _, whisper = self.dispatch()
        whisper.side_effect = RuntimeError("private transcript")
        with self.assertRaises(HTTPException) as error:
            server._dispatch_transcription("parakeet", "/audio", "es", 0)
        self.assertNotIn("private", error.exception.detail)

    def test_v53_single_residency_unloads_parakeet_before_whisper(self):
        _, whisper = self.dispatch(single=True)
        whisper.side_effect = lambda *a: (server._stop_parakeet_worker.assert_called_once(), (["Dolor."], {}))[1]
        server._dispatch_transcription("parakeet", "/audio", "es", 0)

    def test_v53_nonwarm_fallback_releases_whisper(self):
        self.dispatch()
        with patch.object(server, "ASR_WHISPER_KEEP_WARM", False):
            server._dispatch_transcription("parakeet", "/audio", "es", 0)
        server.unload_model.assert_called_once()

    def test_v53_endpoint_archives_both_recognitions_and_cleans_recovered_text(self):
        self.dispatch()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.multiple(server, ASR_BEARER_TOKEN="test-token", ASR_PARAKEET_ENABLED=True,
                           ASR_AUDIO_HISTORY_ENABLED=True, ASR_AUDIO_HISTORY_DIR=directory,
                           ASR_STOP_LLM_FOR_TRANSCRIPTION=False,
                           ASR_PARAKEET_STOP_LLM_FOR_TRANSCRIPTION=False),
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
        self.assertEqual(cleanup.call_args.args[0], "Dolor.")
        self.assertEqual(record["initial_raw_text"], "Донор.")
        self.assertEqual(record["raw_text"], "Dolor.")
        self.assertEqual(record["recognition_engine"], "whisper")
        self.assertEqual(record["fallback_reason"], "spanish_cyrillic")
        self.assertNotIn("Донор", logs.getvalue())
        self.assertNotIn("Dolor", logs.getvalue())
