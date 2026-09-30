import unittest
from unittest.mock import patch

import asr_server


class SpokenNumbersAndSpellingTest(unittest.TestCase):
    def postprocess(self, raw, model_output):
        result = {"choices": [{"message": {"content": model_output}}], "usage": {}}
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", return_value=result),
        ):
            return asr_server._postprocess_transcript(raw)

    def test_v20_spoken_version_number_survives_cleanup(self):
        self.assertEqual(
            self.postprocess("cuatro punto cincuenta y ocho punto cero", "4.58.0"),
            "4.58.0",
        )

    def test_v20_wrong_number_is_rejected(self):
        raw = "versión cuatro punto cincuenta y ocho punto cero"
        self.assertEqual(self.postprocess(raw, "Versión 4.59.0"), raw)

    def test_v20_spoken_hundreds_and_english_numbers(self):
        self.assertEqual(
            self.postprocess("doscientos euros", "200 euros"),
            "200 euros",
        )
        self.assertEqual(
            self.postprocess("four point fifty eight point zero", "4.58.0"),
            "4.58.0",
        )

    def test_v20_number_in_sentence_keeps_other_words(self):
        raw = "instala versión cuatro punto cincuenta y ocho punto cero mañana"
        self.assertEqual(
            self.postprocess(raw, "Instala versión 4.58.0 mañana."),
            "Instala versión 4.58.0 mañana.",
        )
        self.assertEqual(self.postprocess(raw, "Instala versión 4.58.0 hoy."), raw)

    def test_v26_code_digit_sequence_and_hyphen_survive_cleanup(self):
        raw = "aparece un error con error code xyz guión nueve nueve nueve ocho"
        expected = "Aparece un error con error code xyz-9998."
        self.assertEqual(self.postprocess(raw, expected), expected)

    def test_v26_code_digits_are_prepared_even_when_model_keeps_words(self):
        raw = "error code xyz guión nueve nueve nueve ocho"
        self.assertEqual(self.postprocess(raw, raw), "error code xyz-9998")

    def test_v26_code_hyphen_is_restored_from_space_only_output(self):
        raw = "error code xyz guión nueve nueve nueve ocho"
        self.assertEqual(self.postprocess(raw, "Error code xyz 9998."), "Error code xyz-9998.")

    def test_v26_code_digits_preserve_leading_zeros_and_english(self):
        for raw, expected in (
            ("código xyz guion cero cero uno dos", "código xyz-0012"),
            ("error code xyz dash zero zero one two", "error code xyz-0012"),
            ("xyz guion nueve nueve nueve ocho", "xyz-9998"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(self.postprocess(raw, expected), expected)

    def test_v26_wrong_code_digits_and_surrounding_words_are_rejected(self):
        raw = "error code xyz guion cero cero uno dos mañana"
        for output in ("error code xyz-0013 mañana", "error code xyz-12 mañana", "error code xyz-0012 hoy"):
            with self.subTest(output=output):
                self.assertEqual(self.postprocess(raw, output), "error code xyz-0012 mañana")

    def test_v26_ordinary_digit_sequences_are_not_joined(self):
        for raw in ("cuento uno dos tres", "dijo guion nueve nueve veces", "el código tiene nueve nueve ocho", "código xyz guion treinta y dos", "code xyz dash one two hundred"):
            with self.subTest(raw=raw):
                self.assertEqual(self.postprocess(raw, raw), raw)

    def test_v21_spelling_joins_lowercase_in_sentence(self):
        raw = "escribe deletreo eme a erre i a fin de deletreo aquí"
        self.assertEqual(asr_server._prepare_spelled_words(raw), "escribe maria aquí")
        self.assertEqual(self.postprocess(raw, "Escribe maria aquí."), "Escribe maria aquí.")

    def test_v21_spelling_preserves_enye_and_following_words(self):
        self.assertEqual(
            asr_server._prepare_spelled_words("deletreo pe e eñe a fin de deletreo después"),
            "peña después",
        )

    def test_v21_explicit_accent_is_preserved(self):
        self.assertEqual(
            asr_server._prepare_spelled_words(
                "deletreo eme a erre i con tilde a fin de deletreo"
            ),
            "maría",
        )

    def test_v21_english_letter_names_and_initials(self):
        self.assertEqual(
            asr_server._prepare_spelled_words(
                "write spell em ay ar eye ay end spelling here"
            ),
            "write maria here",
        )
        self.assertEqual(
            self.postprocess("write spell M A R I A end spelling here", "Write MARIA here."),
            "Write maria here.",
        )
        self.assertEqual(
            asr_server._prepare_spelled_words("spell double you eye en end spelling"),
            "win",
        )

    def test_v21_prompt_preserves_english_spelling_context(self):
        self.assertIn("Raw: write maria here\nClean: Write Maria here.", asr_server.DEFAULT_POSTPROCESS_PROMPT)

    def test_v21_incomplete_or_unclear_spelling_remains_raw(self):
        for raw in ("deletreo eme a erre", "deletreo eme palabra fin de deletreo"):
            with self.subTest(raw=raw):
                self.assertEqual(asr_server._prepare_spelled_words(raw), raw)

    def test_v21_wrong_spelling_is_rejected(self):
        raw = "escribe deletreo eme a erre i a fin de deletreo aquí"
        self.assertEqual(self.postprocess(raw, "Escribe mario aquí."), "escribe maria aquí")

    def test_v21_block_capitals_are_reduced(self):
        raw = "escribe deletreo eme a erre i a fin de deletreo aquí"
        self.assertEqual(self.postprocess(raw, "Escribe MARIA aquí."), "Escribe maria aquí.")

    def test_v21_spelling_inside_list_item(self):
        raw = "comprar lista de ítems pan nuevo ítem deletreo pe e eñe a fin de deletreo"
        self.assertEqual(
            self.postprocess(raw, "Comprar:\n\n- Pan\n- Peña"),
            "Comprar:\n\n- Pan\n- Peña",
        )


if __name__ == "__main__":
    unittest.main()
