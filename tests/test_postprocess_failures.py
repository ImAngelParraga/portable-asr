import unittest
from unittest.mock import patch

import asr_server
from asr_server import PostprocessUnavailable, _postprocess_transcript


class PostprocessFailureTest(unittest.TestCase):
    def _chat_result(self, content):
        return {
            "choices": [
                {
                    "message": {
                        "content": content,
                    },
                },
            ],
            "usage": {},
        }

    def _postprocess_with_model_output(self, raw, model_output):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(
                asr_server,
                "_llm_chat",
                return_value=self._chat_result(model_output),
            ),
        ):
            return _postprocess_transcript(raw)

    def test_audio_style_postprocess_fails_open_to_raw_text(self):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", side_effect=RuntimeError("llama unavailable")),
        ):
            self.assertEqual(_postprocess_transcript("hello world"), "hello world")

    def test_text_postprocess_can_fail_closed(self):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", side_effect=RuntimeError("llama unavailable")),
        ):
            with self.assertRaises(PostprocessUnavailable):
                _postprocess_transcript("hello world", fail_closed=True)

    def test_obvious_missing_auxiliary_cleanup_is_allowed(self):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(
                asr_server,
                "_llm_chat",
                return_value=self._chat_result("Ramón me ha dicho que no le gusta tu actitud."),
            ),
        ):
            self.assertEqual(
                _postprocess_transcript("Ramón me dicho que no le gusta tu actitud"),
                "Ramón me ha dicho que no le gusta tu actitud.",
            )

    def test_broad_attribution_rewrite_fails_open_to_raw_text(self):
        raw = "Ramón me dicho que no le gusta tu actitud"
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(
                asr_server,
                "_llm_chat",
                return_value=self._chat_result("Me han dicho Ramón que no te gusta la actitud tuya."),
            ),
        ):
            self.assertEqual(_postprocess_transcript(raw), raw)

    def test_pronoun_rewrite_restores_raw_pronouns(self):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(
                asr_server,
                "_llm_chat",
                return_value=self._chat_result("Ramón me ha dicho que no te gusta tu actitud."),
            ),
        ):
            self.assertEqual(
                _postprocess_transcript("Ramón me dicho que no le gusta tu actitud"),
                "Ramón me ha dicho que no le gusta tu actitud.",
            )

    def test_discourse_marker_rewrite_restores_raw_marker(self):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(
                asr_server,
                "_llm_chat",
                return_value=self._chat_result("Sí, tienes razón."),
            ),
        ):
            self.assertEqual(
                _postprocess_transcript("Ya, tienes razón"),
                "Ya, tienes razón.",
            )

    def test_v11_prompt_contains_bilingual_symbols_and_cues(self):
        prompt = asr_server.DEFAULT_POSTPROCESS_PROMPT
        self.assertIn("hyphen / dash / guion / guión -> -", prompt)
        self.assertIn("underscore / guion bajo / guión bajo / barra baja -> _", prompt)
        for cue in (
            "class",
            "clase",
            "function",
            "función",
            "method",
            "método",
            "folder",
            "carpeta",
            "directory",
            "directorio",
            "path",
            "ruta",
            "document",
            "documento",
        ):
            with self.subTest(cue=cue):
                self.assertIn(cue, prompt)

    def test_v11_spanish_barra_baja_survives_full_pipeline(self):
        self.assertEqual(
            self._postprocess_with_model_output(
                "la carpeta Debian barra baja Developer barra baja Install",
                "La carpeta Debian_Developer_Install.",
            ),
            "La carpeta debian_developer_install.",
        )

    def test_v11_spanish_class_space_only_hyphen_reconstructed(self):
        self.assertEqual(
            self._postprocess_with_model_output(
                "la clase Debian guion Developer guion Install",
                "La clase Debian Developer Install.",
            ),
            "La clase debian-developer-install.",
        )

    def test_v11_english_class_space_only_hyphen_reconstructed(self):
        self.assertEqual(
            self._postprocess_with_model_output(
                "the class Debian dash Developer dash Install",
                "The class Debian Developer Install.",
            ),
            "The class debian-developer-install.",
        )

    def test_v11_bilingual_function_and_method_cues(self):
        cases = (
            (
                "la función parse guion token",
                "La función parse token.",
                "La función parse-token.",
            ),
            (
                "el método parse guion bajo token",
                "El método parse token.",
                "El método parse_token.",
            ),
            (
                "the function parse hyphen token",
                "The function parse token.",
                "The function parse-token.",
            ),
            (
                "the method parse underscore token",
                "The method parse token.",
                "The method parse_token.",
            ),
        )
        for raw, model_output, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    self._postprocess_with_model_output(raw, model_output),
                    expected,
                )

    def test_v11_ordinary_barra_baja_sentence_preserved(self):
        self.assertEqual(
            self._postprocess_with_model_output(
                "la barra baja está mal",
                "La barra_baja está mal.",
            ),
            "La barra baja está mal.",
        )

    def test_v11_ordinary_english_hyphen_sentence_preserved(self):
        self.assertEqual(
            self._postprocess_with_model_output(
                "the hyphen is wrong",
                "The-is wrong.",
            ),
            "The hyphen is wrong.",
        )


if __name__ == "__main__":
    unittest.main()
