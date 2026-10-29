import ctypes
import io
import os
import threading
import time
from unittest.mock import Mock, patch
import unittest

import asr_server as server
import parakeet_asr_worker as worker


class CppWorkerTest(unittest.TestCase):
    def fake_library(self, text=b"Hola"):
        library = Mock()
        library.parakeet_capi_load.return_value = 123
        buffer = ctypes.create_string_buffer(text)
        library.parakeet_capi_transcribe_path.return_value = ctypes.addressof(buffer)
        return library, buffer

    def test_v42_reuses_context_and_frees_strings_and_context(self):
        library, buffer = self.fake_library()
        with patch.object(worker.ctypes, "CDLL", return_value=library):
            model = worker.CppParakeet("/runtime/libparakeet.so", "/models/ultra.gguf")
            for _ in range(2):
                self.assertEqual(model.transcribe(audio="/audio.wav"), {"text": "Hola"})
            model.close()
            model.close()
        library.parakeet_capi_load.assert_called_once_with(b"/models/ultra.gguf")
        self.assertEqual(library.parakeet_capi_free_string.call_count, 2)
        library.parakeet_capi_free.assert_called_once_with(123)

    def test_v42_decode_failure_still_frees_native_string(self):
        library, buffer = self.fake_library(b"\xff")
        with patch.object(worker.ctypes, "CDLL", return_value=library):
            model = worker.CppParakeet("/runtime/libparakeet.so", "/ultra.gguf")
            with self.assertRaises(UnicodeDecodeError):
                model.transcribe(audio="/audio.wav")
            library.parakeet_capi_free_string.assert_called_once()
            model.close()

    def test_v42_null_context_and_null_transcript_fail_safely(self):
        library, buffer = self.fake_library()
        with patch.object(worker.ctypes, "CDLL", return_value=library):
            library.parakeet_capi_load.return_value = None
            with self.assertRaisesRegex(RuntimeError, "Could not load"):
                worker.CppParakeet("/runtime/libparakeet.so", "/ultra.gguf")
            library.parakeet_capi_load.return_value = 123
            model = worker.CppParakeet("/runtime/libparakeet.so", "/ultra.gguf")
            library.parakeet_capi_transcribe_path.return_value = None
            with self.assertRaisesRegex(RuntimeError, "transcription failed"):
                model.transcribe(audio="/audio.wav")
            library.parakeet_capi_free_string.assert_not_called()
            model.close()

    def test_v42_cpp_environment_excludes_whisper_and_selects_library(self):
        with (
            patch.multiple(server, _parakeet_worker_process=None,
                           ASR_PARAKEET_BACKEND="cpp", ASR_PARAKEET_LIBRARY="/runtime/libparakeet.so",
                           ASR_PARAKEET_GGUF="/models/ultra.gguf"),
            patch.dict(os.environ, {"LD_LIBRARY_PATH": "/whisper/cudnn"}),
            patch.object(server.subprocess, "Popen") as popen,
        ):
            server._start_parakeet_worker_locked()
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["LD_LIBRARY_PATH"], "/runtime")
            self.assertEqual(env["ASR_PARAKEET_GGUF"], "/models/ultra.gguf")
            self.assertEqual(env["ASR_PARAKEET_BACKEND"], "cpp")
            self.assertEqual(env["PARAKEET_DEVICE"], "CUDA0")

    def test_v42_cpp_selection_does_not_import_photon(self):
        with (
            patch.dict(os.environ, {"ASR_PARAKEET_BACKEND": "cpp", "ASR_PARAKEET_LIBRARY": "/lib.so", "ASR_PARAKEET_GGUF": "/ultra.gguf"}),
            patch.object(worker, "CppParakeet", return_value="native-model") as model,
        ):
            self.assertEqual(worker.load_model(), "native-model")
            model.assert_called_once_with("/lib.so", "/ultra.gguf")


class ResidencyTest(unittest.TestCase):
    def test_v43_stops_only_unselected_workers_before_inference(self):
        for selected in ["whisper", "qwen", "parakeet"]:
            events = []
            with (
                self.subTest(selected=selected),
                patch.object(server, "ASR_SINGLE_RESIDENT_MODEL", True),
                patch.object(server, "unload_model", side_effect=lambda: events.append("unload-main")),
                patch.object(server, "_stop_whisper_worker", side_effect=lambda *a: events.append("stop-whisper")),
                patch.object(server, "_stop_qwen_worker", side_effect=lambda *a: events.append("stop-qwen")),
                patch.object(server, "_stop_parakeet_worker", side_effect=lambda *a: events.append("stop-parakeet")),
                patch.object(server, "_run_transcription", side_effect=lambda *a: events.append("whisper") or (["Hola"], {})),
                patch.object(server, "_run_qwen_transcription", side_effect=lambda *a: events.append("qwen") or (["Hola"], {})),
                patch.object(server, "_run_parakeet_transcription", side_effect=lambda *a: events.append("parakeet") or (["Hola"], {})),
                patch.object(server, "_stop_llm_server") as stop_cleanup,
            ):
                for _ in range(2):
                    self.assertEqual(server._dispatch_transcription(selected, "/audio.wav", "es", 0, "vocab"), (["Hola"], {}))
                self.assertEqual(events[-1], selected)
                self.assertNotIn("stop-" + selected, events)
                for other in {"whisper", "qwen", "parakeet"} - {selected}:
                    self.assertIn("stop-" + other, events)
                stop_cleanup.assert_not_called()

    def test_v43_disabled_policy_preserves_independent_workers(self):
        with (
            patch.object(server, "ASR_SINGLE_RESIDENT_MODEL", False),
            patch.object(server, "_run_parakeet_transcription", return_value=(["Hola"], {})),
            patch.object(server, "_stop_whisper_worker") as whisper,
            patch.object(server, "_stop_qwen_worker") as qwen,
        ):
            server._dispatch_transcription("parakeet", "/audio.wav", None, 0)
            whisper.assert_not_called()
            qwen.assert_not_called()

    def test_v43_switch_waits_for_active_inference(self):
        started = threading.Event()
        release = threading.Event()
        second_done = threading.Event()
        events = []

        def first(*args):
            events.append("first-start")
            started.set()
            release.wait(3)
            events.append("first-end")
            return ["first"], {}

        def second(*args):
            events.append("second-start")
            second_done.set()
            return ["second"], {}

        with (
            patch.object(server, "ASR_SINGLE_RESIDENT_MODEL", True),
            patch.object(server, "unload_model"),
            patch.object(server, "_stop_whisper_worker"),
            patch.object(server, "_stop_qwen_worker"),
            patch.object(server, "_stop_parakeet_worker"),
            patch.object(server, "_run_qwen_transcription", side_effect=first),
            patch.object(server, "_run_parakeet_transcription", side_effect=second),
        ):
            a = threading.Thread(target=server._dispatch_transcription, args=("qwen", "/a.wav", None, 0))
            b = threading.Thread(target=server._dispatch_transcription, args=("parakeet", "/b.wav", None, 0))
            a.start()
            try:
                self.assertTrue(started.wait(1))
                b.start()
                self.assertFalse(second_done.wait(.05))
            finally:
                release.set()
                a.join(3)
                if b.ident is not None:
                    b.join(3)
            self.assertEqual(events, ["first-start", "first-end", "second-start"])


if __name__ == "__main__":
    unittest.main()
