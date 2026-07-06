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


if __name__ == "__main__":
    unittest.main()
