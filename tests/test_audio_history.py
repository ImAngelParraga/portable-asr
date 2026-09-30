import contextlib
import io
import json
import stat
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import asr_server


class AudioHistoryTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(temporary) / "history"
        self.stack.enter_context(patch.multiple(
            asr_server, create=True,
            ASR_AUDIO_HISTORY_ENABLED=True,
            ASR_AUDIO_HISTORY_DIR=str(self.root),
            ASR_AUDIO_HISTORY_MAX_FILES=20,
            ASR_AUDIO_HISTORY_MAX_AGE_SECONDS=86400,
            ASR_AUDIO_HISTORY_MAX_BYTES=256 * 1024 * 1024,
            ASR_BEARER_TOKEN="test-token",
            ASR_STOP_LLM_FOR_TRANSCRIPTION=False,
            ASR_WHISPER_KEEP_WARM=True,
        ))
        self.stack.enter_context(patch.object(asr_server, "_run_transcription", return_value=(["raw question"], {})))
        self.stack.enter_context(patch.object(asr_server, "_postprocess_transcript", return_value="Final question?"))
        self.client = TestClient(asr_server.app)

    def upload(self, content=b"original-upload-bytes", **kwargs):
        return self.client.post(
            "/v1/audio/transcriptions",
            headers=kwargs.pop("headers", {"Authorization": "Bearer test-token"}),
            data={"model": "whisper-1", "language": "es", "prompt": "PrivateVocabulary"},
            files={"file": ("private-filename.wav", content, kwargs.pop("content_type", "audio/wav"))},
            **kwargs,
        )

    def records(self):
        return sorted(self.root.glob("*/metadata.json"))

    def start_at(self, timestamp, content=b"audio"):
        with (
            patch.object(asr_server.time, "time_ns", return_value=int(timestamp * 1e9)),
            patch.object(asr_server.time, "time", return_value=timestamp),
        ):
            return asr_server._audio_history_start(content, ".wav", {"engine": "whisper"})

    def test_v24_capture_preserves_bytes_raw_final_and_request_id(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            response = self.upload()
        self.assertEqual(response.status_code, 200)
        record = json.loads(self.records()[0].read_text())
        self.assertEqual(record["request_id"], response.json()["request_id"])
        self.assertEqual(record["raw_text"], "raw question")
        self.assertEqual(record["final_text"], "Final question?")
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["engine"], "whisper")
        self.assertEqual(record["language"], "es")
        self.assertEqual(record["prompt_chars"], len("PrivateVocabulary"))
        self.assertEqual((self.records()[0].parent / record["audio_file"]).read_bytes(), b"original-upload-bytes")
        serialized = self.records()[0].read_text()
        for private in ("test-token", "PrivateVocabulary", "private-filename"):
            self.assertNotIn(private, serialized)
            self.assertNotIn(private, output.getvalue())
        self.assertNotIn("raw question", output.getvalue())
        self.assertNotIn("Final question?", output.getvalue())

    def test_v24_qwen_capture_uses_selected_engine(self):
        with (
            patch.object(asr_server, "ASR_QWEN_ENABLED", True),
            patch.object(asr_server, "_run_qwen_transcription", return_value=(["qwen raw"], {})),
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer test-token"},
                data={"model": asr_server.ASR_QWEN_MODEL_ID, "language": "es"},
                files={"file": ("input.m4a", b"aac-bytes", "audio/mp4")},
            )
        self.assertEqual(response.status_code, 200)
        record = json.loads(self.records()[0].read_text())
        self.assertEqual(record["engine"], "qwen")
        self.assertEqual(record["raw_text"], "qwen raw")
        self.assertEqual(record["audio_file"], "audio.m4a")

    def test_v24_disabled_history_preserves_response_and_creates_nothing(self):
        with patch.object(asr_server, "ASR_AUDIO_HISTORY_ENABLED", False):
            response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("request_id", response.json())
        self.assertFalse(self.root.exists())

    def test_v24_failed_asr_retains_audio_and_failure_id(self):
        with patch.object(asr_server, "_run_transcription", side_effect=RuntimeError("decode failed")):
            response = self.upload()
        self.assertEqual(response.status_code, 500)
        record = json.loads(self.records()[0].read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertEqual(record["request_id"], response.headers["X-Request-ID"])
        self.assertIsNone(record["raw_text"])
        self.assertIsNone(record["final_text"])
        self.assertNotIn("decode failed", self.records()[0].read_text())

    def test_v24_failed_cleanup_retains_raw_asr(self):
        with patch.object(asr_server, "_postprocess_transcript", side_effect=asr_server.PostprocessUnavailable("cleanup failed")):
            response = self.upload()
        self.assertEqual(response.status_code, 500)
        record = json.loads(self.records()[0].read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["raw_text"], "raw question")
        self.assertIsNone(record["final_text"])

    def test_v25_private_permissions(self):
        self.upload()
        metadata = self.records()[0]
        for directory in (self.root, metadata.parent):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        for file in metadata.parent.iterdir():
            self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)

    def test_v25_oldest_records_evicted_by_count(self):
        with patch.object(asr_server, "ASR_AUDIO_HISTORY_MAX_FILES", 2):
            old = self.start_at(1000)
            recent = self.start_at(1001)
            newest = self.start_at(1002)
        ids = {json.loads(path.read_text())["request_id"] for path in self.records()}
        self.assertEqual(ids, {recent["request_id"], newest["request_id"]})
        self.assertNotIn(old["request_id"], ids)

    def test_v25_expiry_without_new_upload(self):
        self.start_at(1000)
        with patch.object(asr_server.time, "time", return_value=1000 + 86401):
            asr_server._audio_history_prune()
        self.assertEqual(self.records(), [])

    def test_v25_idle_watcher_prunes_on_startup_and_every_minute(self):
        from unittest.mock import Mock
        stop = Mock()
        stop.is_set.return_value = False
        stop.wait.side_effect = [False, True]
        with patch.object(asr_server, "_audio_history_prune") as prune:
            asr_server._audio_history_watcher_loop(stop)
        self.assertEqual(prune.call_count, 2)
        self.assertEqual([call.args for call in stop.wait.call_args_list], [(60,), (60,)])

    def test_v25_lifespan_starts_and_stops_history_watcher(self):
        with (
            patch.object(asr_server, "_gpu_watch_enabled", return_value=False),
            patch.object(asr_server.threading, "Thread") as thread,
            patch.object(asr_server, "_stop_whisper_worker"),
            patch.object(asr_server, "_stop_qwen_worker"),
            patch.object(asr_server, "unload_model"),
            patch.object(asr_server, "_stop_llm_server"),
        ):
            with TestClient(asr_server.app):
                thread.assert_called_once()
                self.assertIs(thread.call_args.kwargs["target"], asr_server._audio_history_watcher_loop)
                stop = thread.call_args.kwargs["args"][0]
                self.assertFalse(stop.is_set())
                thread.return_value.start.assert_called_once()
            self.assertTrue(stop.is_set())

    def test_v25_byte_limit_and_oversized_single_record(self):
        with patch.object(asr_server, "ASR_AUDIO_HISTORY_MAX_BYTES", 1800):
            self.start_at(1000, b"a" * 900)
            newest = self.start_at(1001, b"b" * 900)
            self.assertEqual(len(self.records()), 1)
            self.assertEqual(json.loads(self.records()[0].read_text())["request_id"], newest["request_id"])
            self.assertIsNone(self.start_at(1002, b"c" * 2000))
        total = sum(file.stat().st_size for path in self.records() for file in path.parent.iterdir())
        self.assertLessEqual(total, 1800)

    def test_v25_storage_failure_does_not_fail_transcription(self):
        self.root.write_text("not a directory")
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Final question?")
        self.assertNotIn("request_id", response.json())

    def test_v25_metadata_failure_does_not_fail_transcription(self):
        with patch.object(asr_server, "_audio_history_update", side_effect=OSError("disk full")):
            response = self.upload()
        self.assertEqual(response.status_code, 200)

    def test_v25_rejected_uploads_create_nothing(self):
        self.assertEqual(self.upload(headers={}).status_code, 401)
        self.assertEqual(self.upload(content_type="text/plain").status_code, 400)
        with patch.object(asr_server, "ASR_MAX_FILE_SIZE", 1):
            self.assertEqual(self.upload().status_code, 413)
        self.assertFalse(self.root.exists())

    def test_v25_concurrent_writes_remain_bounded(self):
        with patch.object(asr_server, "ASR_AUDIO_HISTORY_MAX_FILES", 3):
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda _: asr_server._audio_history_start(b"bytes", ".wav", {}), range(8)))
        self.assertEqual(len(self.records()), 3)
        for path in self.records():
            self.assertEqual((path.parent / json.loads(path.read_text())["audio_file"]).read_bytes(), b"bytes")

    def test_v25_symlink_root_and_unrelated_files_untouched(self):
        target = self.root.parent / "outside"
        target.mkdir()
        (target / "keep.txt").write_text("keep")
        self.root.symlink_to(target, target_is_directory=True)
        self.assertIsNone(asr_server._audio_history_start(b"bytes", ".wav", {}))
        self.assertEqual((target / "keep.txt").read_text(), "keep")

    def test_v25_invalid_limits_disable_storage(self):
        for setting in ("ASR_AUDIO_HISTORY_MAX_FILES", "ASR_AUDIO_HISTORY_MAX_AGE_SECONDS", "ASR_AUDIO_HISTORY_MAX_BYTES"):
            with self.subTest(setting=setting), patch.object(asr_server, setting, 0):
                self.assertIsNone(asr_server._audio_history_start(b"bytes", ".wav", {}))
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
