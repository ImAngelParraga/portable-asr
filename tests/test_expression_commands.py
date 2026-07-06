import unittest

from asr_server import (
    _apply_isolated_expression_command,
    _apply_isolated_expression_sentences,
    _protect_meaningful_order_rewrite,
    _protect_non_isolated_expression_rewrite,
    _protect_spanish_discourse_marker_rewrite,
    _protect_spanish_diminutive_rewrite,
    _protect_spanish_pronoun_rewrite,
)


class ExpressionCommandTest(unittest.TestCase):
    def test_isolated_spanish_laughter_commands_convert(self):
        self.assertEqual(_apply_isolated_expression_command("risa"), "jajaja")
        self.assertEqual(_apply_isolated_expression_command("risas."), "jajaja")
        self.assertEqual(_apply_isolated_expression_command("  Risas!  "), "jajaja")

    def test_isolated_english_laughter_command_converts(self):
        self.assertEqual(_apply_isolated_expression_command("laugh"), "hahaha")
        self.assertEqual(_apply_isolated_expression_command("Laugh!"), "hahaha")

    def test_laughter_words_inside_sentences_do_not_convert(self):
        self.assertEqual(_apply_isolated_expression_command("me dio risa"), "me dio risa")
        self.assertEqual(_apply_isolated_expression_command("la risa fue fuerte"), "la risa fue fuerte")
        self.assertEqual(_apply_isolated_expression_command("risa nerviosa"), "risa nerviosa")
        self.assertEqual(_apply_isolated_expression_command("please laugh now"), "please laugh now")

    def test_isolated_laughter_sentence_inside_longer_text_converts(self):
        text = "Estoy probando esto. Risas. Ahora hablo de trabajo."
        self.assertEqual(
            _apply_isolated_expression_sentences(text),
            "Estoy probando esto. jajaja. Ahora hablo de trabajo.",
        )

    def test_laughter_word_inside_longer_sentence_does_not_convert(self):
        text = "Estoy probando esto. Me dio risa. Ahora hablo de trabajo."
        self.assertEqual(_apply_isolated_expression_sentences(text), text)

    def test_laughter_only_rewrite_is_rejected_for_non_isolated_sentence(self):
        self.assertEqual(_protect_non_isolated_expression_rewrite("me dio risa", "jajaja"), "me dio risa")
        self.assertEqual(_protect_non_isolated_expression_rewrite("please laugh now", "hahaha"), "please laugh now")
        self.assertEqual(_protect_non_isolated_expression_rewrite("risas", "jajaja"), "jajaja")

    def test_diminutive_rewrite_is_rejected(self):
        self.assertEqual(
            _protect_spanish_diminutive_rewrite(
                "Esto fue un fallito que vi en la aplicación",
                "Esto fue un fallo que vi en la aplicación.",
            ),
            "Esto fue un fallito que vi en la aplicación",
        )

    def test_diminutive_preserved_allows_cleanup(self):
        self.assertEqual(
            _protect_spanish_diminutive_rewrite(
                "Los mensajes son cortitos",
                "Los mensajes son cortitos.",
            ),
            "Los mensajes son cortitos.",
        )

    def test_missing_auxiliary_insertion_preserves_order(self):
        self.assertEqual(
            _protect_meaningful_order_rewrite(
                "Ramón me dicho que no le gusta tu actitud",
                "Ramón me ha dicho que no le gusta tu actitud.",
            ),
            "Ramón me ha dicho que no le gusta tu actitud.",
        )

    def test_broad_rewrite_with_changed_attribution_is_rejected(self):
        raw = "Ramón me dicho que no le gusta tu actitud"
        self.assertEqual(
            _protect_meaningful_order_rewrite(
                raw,
                "Me han dicho Ramón que no te gusta la actitud tuya.",
            ),
            raw,
        )

    def test_spanish_pronoun_rewrite_is_rejected(self):
        self.assertEqual(
            _protect_spanish_pronoun_rewrite(
                "Ramón me dicho que no le gusta tu actitud",
                "Ramón me ha dicho que no te gusta tu actitud.",
            ),
            "Ramón me ha dicho que no le gusta tu actitud.",
        )

    def test_spanish_discourse_marker_rewrite_is_restored(self):
        self.assertEqual(
            _protect_spanish_discourse_marker_rewrite(
                "Ya, tienes razón",
                "Sí, tienes razón.",
            ),
            "Ya, tienes razón.",
        )

    def test_spanish_discourse_marker_cleanup_is_allowed(self):
        self.assertEqual(
            _protect_spanish_discourse_marker_rewrite(
                "Ya tienes razón",
                "Ya, tienes razón.",
            ),
            "Ya, tienes razón.",
        )


if __name__ == "__main__":
    unittest.main()
