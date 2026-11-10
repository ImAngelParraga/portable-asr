import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import asr_server
from spoken_commands import CommandDictationError, parse_spoken_command


class SpokenCommandTest(unittest.TestCase):
    def test_commands_symbols_spelling_and_case(self):
        examples = {
            "comando barra new": "/new",
            "comando slash clear.": "/clear",
            "modo comando barra new": "/new",
            "Command slash new": "/new",
            "comando deletreo m k d i r": "mkdir",
            "Comando. Deletreo. Eme, ka, de, i, erre.": "mkdir",
            "comando deletreo eme ka de i erre fin deletreo espacio proyectos": "mkdir proyectos",
            "comando deletreo eme ka de i erre fin de deletreo proyectos": "mkdir proyectos",
            "command spell em kay dee eye ar end spelling space projects": "mkdir projects",
            "comando git espacio status": "git status",
            "comando git status": "git status",
            "comando git espacio guion guion version": "git --version",
            "comando cat espacio barra etc barra hosts": "cat /etc/hosts",
            "comando cat espacio punto barra script punto sh": "cat ./script.sh",
            "comando cat script.sh.": "cat script.sh",
            "comando touch espacio mi guion bajo archivo punto txt": "touch mi_archivo.txt",
            "comando echo espacio comillas hello world comillas": 'echo "hello world"',
            "comando echo espacio mayúscula home": "echo HOME",
            "comando deletreo mayúscula eme ka de i erre": "Mkdir",
            "comando deletreo doble u i griega equis": "wyx",
            "command spell double you why ex": "wyx",
            "comando deletreo a cero cero siete": "a007",
            "comando python tres": "python 3",
            "comando echo espacio dólar mayúscula home": "echo $HOME",
        }
        for spoken, expected in examples.items():
            with self.subTest(spoken=spoken):
                self.assertEqual(parse_spoken_command(spoken), expected)

    def test_ordinary_prose_is_not_command_dictation(self):
        for text in ("", "quiero usar el comando mkdir", "comandos de Linux", "lista con viñetas pan siguiente elemento leche"):
            with self.subTest(text=text):
                self.assertIsNone(parse_spoken_command(text))

    def test_malformed_controls_never_guess_a_command(self):
        for text in (
            "comando", "comando deletreo", "comando deletreo eme desconocida",
            "comando deletreo eme ka espacio proyectos", "comando fin deletreo",
            "comando mayúscula", "comando mayúscula slash new", "comando espacio",
            "comando deletreo eme mayúscula", "comando mayúscula mayúscula home",
        ):
            with self.subTest(text=text):
                with self.assertRaises(CommandDictationError):
                    parse_spoken_command(text)

    def test_command_bypasses_llm_with_cleanup_enabled_or_disabled(self):
        for enabled in (True, False):
            with (
                self.subTest(enabled=enabled),
                patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", enabled),
                patch.object(asr_server, "_llm_chat") as llm,
            ):
                self.assertEqual(asr_server._postprocess_transcript("comando slash new"), "/new")
                self.assertEqual(
                    asr_server._postprocess_transcript("comando slash new", "Rewrite this text", True, True),
                    "/new",
                )
                llm.assert_not_called()

    def test_invalid_spelling_is_recoverable_error_without_llm(self):
        with patch.object(asr_server, "_llm_chat") as llm:
            with self.assertRaises(HTTPException) as error:
                asr_server._postprocess_transcript("comando deletreo eme desconocida")
            self.assertEqual(error.exception.status_code, 422)
            llm.assert_not_called()

    def test_command_mode_does_not_leak_to_next_recording(self):
        with patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", False):
            self.assertEqual(asr_server._postprocess_transcript("comando slash new"), "/new")
            self.assertEqual(asr_server._postprocess_transcript("barra new"), "barra new")

    def _audio_response(self, text):
        with (
            patch.object(asr_server, "ASR_BEARER_TOKEN", "test-token"),
            patch.object(asr_server, "ASR_STOP_LLM_FOR_TRANSCRIPTION", False),
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", False),
            patch.object(asr_server, "ASR_AUDIO_HISTORY_ENABLED", False),
            patch.object(asr_server, "_dispatch_transcription", return_value=([text], {})),
            patch.object(asr_server, "unload_model"),
        ):
            return TestClient(asr_server.app).post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer test-token"},
                data={"model": "whisper-1"},
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
            )

    def test_audio_response_marks_only_commands_verbatim(self):
        command = self._audio_response("comando slash new")
        self.assertEqual(command.status_code, 200)
        self.assertEqual(command.json()["text"], "/new")
        self.assertIs(command.json()["verbatim"], True)
        prose = self._audio_response("normal dictation")
        self.assertEqual(prose.json()["text"], "normal dictation")
        self.assertNotIn("verbatim", prose.json())

    def test_audio_rejects_invalid_spelling_without_returning_command(self):
        response = self._audio_response("comando deletreo eme desconocida")
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("text", response.json())
