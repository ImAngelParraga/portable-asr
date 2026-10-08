import unittest
from unittest.mock import patch

import asr_server


class SpeechRestartTest(unittest.TestCase):
    def cleanup(self, raw, output):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", return_value={
                "choices": [{"message": {"content": output}}], "usage": {},
            }) as chat,
        ):
            result = asr_server._postprocess_transcript(raw)
        return result, chat.call_args.args[0][-1]["content"]

    def test_v34_corrected_prepositions_survive_full_cleanup(self):
        cases = (
            ("Quiero ir a, al rocódromo de allí", "Quiero ir al rocódromo de allí"),
            ("Quiero ir a... al rocódromo de allí", "Quiero ir al rocódromo de allí"),
            ("You can see that on, in this file", "You can see that in this file"),
            ("You can see that on… in this file", "You can see that in this file"),
            ("You can see that in — on this page", "You can see that on this page"),
            ("Voy a, al rocódromo a las 5", "Voy al rocódromo a las 5"),
            ("Vengo de, del rocódromo", "Vengo del rocódromo"),
            ("Voy a, a casa", "Voy a casa"),
            ("Put it in, in this file", "Put it in this file"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                result, prompt = self.cleanup(raw, expected + ".")
                self.assertEqual(result, expected + ".")
                self.assertIn(expected, prompt)
                self.assertEqual(self.cleanup(raw, raw)[0], expected)

    def test_v34_meaningful_and_ambiguous_speech_remains_literal(self):
        cases = (
            "Put it on or in this file",
            "Put it on, in, or under the box",
            "Put it on, in or under the box",
            "Use prepositions on, in this example",
            'He said "put it on, in this file" yesterday',
            "Use `on, in` here",
            "Use 'on, in this file' here",
            "You can see that on this page, in this file",
            "Quiero ir a Madrid, al rocódromo de allí",
            "Quiero ir a al rocódromo de allí",
            "No, no quiero ir",
            "Muy, muy bien",
            "Quiero ir mañana, no, el viernes",
            "spell on, in end spelling",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(self.cleanup(raw, raw)[0], raw)

    def test_v34_list_controls_keep_their_items(self):
        self.assertEqual(self.cleanup("Item list on, in end of list", "- on\n- in")[0],
                         "- on\n- in")

    def test_v34_disabled_and_unavailable_cleanup_returns_raw(self):
        raw = "Voy a, al rocódromo"
        with patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", False):
            self.assertEqual(asr_server._postprocess_transcript(raw), raw)
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", side_effect=RuntimeError("unavailable")),
        ):
            self.assertEqual(asr_server._postprocess_transcript(raw), raw)
            with self.assertRaises(asr_server.PostprocessUnavailable):
                asr_server._postprocess_transcript(raw, fail_closed=True)

    def test_v34_other_content_changes_are_rejected(self):
        raw = "Voy a, al rocódromo a las 5"
        self.assertEqual(self.cleanup(raw, "Voy al rocódromo a las 6.")[0],
                         "Voy al rocódromo a las 5")
