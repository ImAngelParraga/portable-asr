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

    def test_v56_phrase_restart_keeps_last_choice(self):
        cases = (
            ("Para la ausencia médica puedes utilizar esta tarea dentro del tiempo, dentro del tempo.",
             "Para la ausencia médica puedes utilizar esta tarea dentro del tempo."),
            ("Dentro del tiempo... dentro del tempo", "Dentro del tempo"),
            ("Dentro del tempo… dentro del tiempo", "Dentro del tiempo"),
            ("Regístralo dentro del tiempo — dentro del tempo hoy.",
             "Regístralo dentro del tempo hoy."),
        )
        for raw, expected in cases:
            for output in (raw, expected):
                with self.subTest(raw=raw, output=output):
                    result, prompt = self.cleanup(raw, output)
                    self.assertEqual(result, expected)
                    self.assertIn(expected, prompt)

    def test_v56_phrase_restart_preserves_meaningful_repetitions(self):
        cases = (
            "Dentro del tiempo, dentro del tiempo.",
            "Dentro del tiempo dentro del tempo.",
            "Dentro del tiempo, dentro del espacio.",
            "Mañana lo mando, mañana lo mandó.",
            "En el mando, en el mandó.",
            "En los sesenta, en los setenta.",
            "We should walk, we should talk.",
            "Cuesta unos veinte, cuesta unos treinta.",
            "Dentro del tiempo, dentro del tempo y dentro del templo.",
            'Dijo "dentro del tiempo, dentro del tempo".',
            "Usa `dentro del tiempo, dentro del tempo`.",
            "La frase es dentro del tiempo, dentro del tempo.",
            "Lista de ítems dentro del tiempo, dentro del tempo fin de lista",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(asr_server._prepare_speech_restarts(raw), raw)

    def test_v56_phrase_restart_failure_contract(self):
        raw = "Dentro del tiempo, dentro del tempo."
        with patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", False):
            self.assertEqual(asr_server._postprocess_transcript(raw), raw)
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", side_effect=RuntimeError("unavailable")),
        ):
            self.assertEqual(asr_server._postprocess_transcript(raw), raw)
            with self.assertRaises(asr_server.PostprocessUnavailable):
                asr_server._postprocess_transcript(raw, fail_closed=True)

    def test_v56_model_cannot_restore_abandoned_final_word(self):
        raw = "Para la ausencia médica puedes utilizar esta tarea dentro del tiempo, dentro del tempo."
        edited = "Para la ausencia médica puedes utilizar esta tarea dentro del tiempo."
        expected = "Para la ausencia médica puedes utilizar esta tarea dentro del tempo."
        self.assertEqual(self.cleanup(raw, edited)[0], expected)

    def test_v34_corrected_prepositions_survive_full_cleanup(self):
        cases = (
            ("Quiero ir a, al edificio de allí", "Quiero ir al edificio de allí"),
            ("Quiero ir a... al edificio de allí", "Quiero ir al edificio de allí"),
            ("You can see that on, in this file", "You can see that in this file"),
            ("You can see that on… in this file", "You can see that in this file"),
            ("You can see that in — on this page", "You can see that on this page"),
            ("Voy a, al edificio a las 5", "Voy al edificio a las 5"),
            ("Vengo de, del edificio", "Vengo del edificio"),
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
            "Quiero ir a Madrid, al edificio de allí",
            "Quiero ir a al edificio de allí",
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
        raw = "Voy a, al edificio"
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
        raw = "Voy a, al edificio a las 5"
        self.assertEqual(self.cleanup(raw, "Voy al edificio a las 6.")[0],
                         "Voy al edificio a las 5")
