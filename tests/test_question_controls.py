import unittest
from unittest.mock import patch

import asr_server


class QuestionControlsTest(unittest.TestCase):
    def cleanup(self, raw, model_output):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", return_value={
                "choices": [{"message": {"content": model_output}}],
            }) as chat,
        ):
            result = asr_server._postprocess_transcript(raw)
        return result, chat.call_args.args[0][-1]["content"]

    def test_v30_mid_transcript_command_survives_statement_output(self):
        result, prompt = self.cleanup(
            "Ya terminé. Pregunta puedes revisarlo. Luego hablamos.",
            "Ya terminé. Puedes revisarlo. Luego hablamos.",
        )
        self.assertEqual(result, "Ya terminé. ¿Puedes revisarlo? Luego hablamos.")
        self.assertIn("¿Puedes revisarlo?", prompt)

    def test_v30_repeated_commands_and_identical_unmarked_sentences(self):
        result, _ = self.cleanup(
            "Vienes mañana. Pregunta vienes mañana. Pregunta puedes venir. Puedes venir.",
            "Vienes mañana. Vienes mañana. Puedes venir. Puedes venir.",
        )
        self.assertEqual(result, "Vienes mañana. ¿Vienes mañana? ¿Puedes venir? Puedes venir.")

    def test_v30_line_break_and_period_after_command(self):
        result, _ = self.cleanup(
            "Ya terminé.\nPregunta. Puedes revisarlo.\nPregunta: cuándo vienes.",
            "Ya terminé.\nPuedes revisarlo.\nCuándo vienes.",
        )
        self.assertEqual(result, "Ya terminé.\n¿Puedes revisarlo?\n¿Cuándo vienes?")

    def test_v30_model_cannot_merge_following_statement_into_question(self):
        result, _ = self.cleanup(
            "Pregunta puedes revisarlo. Luego hablamos.",
            "¿Puedes revisarlo luego hablamos?",
        )
        self.assertEqual(result, "¿Puedes revisarlo? Luego hablamos.")

    def test_v30_guard_fallback_keeps_every_question(self):
        result, _ = self.cleanup(
            "Ya terminé. Pregunta ahora sí. Pregunta vienes mañana.",
            "Ya terminé. ¿Por qué ahora sí? ¿Vienes mañana?",
        )
        self.assertEqual(result, "Ya terminé. ¿Ahora sí? ¿Vienes mañana?")

    def test_v30_ordinary_word_and_empty_command_stay_literal(self):
        for raw in ("Tengo una pregunta para ti.", "La pregunta es clara.", "pregunta", "pregunta ¿?"):
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, raw)
                self.assertEqual(result, raw)

    def test_v30_model_cannot_move_either_question_boundary(self):
        cases = (
            ("Ya terminé. Pregunta puedes revisarlo.", "Ya terminé ¿Puedes revisarlo?", "Ya terminé. ¿Puedes revisarlo?"),
            ("Pregunta puedes revisarlo. Luego hablamos.", "¿Puedes revisarlo. Luego hablamos?", "¿Puedes revisarlo? Luego hablamos."),
            ("Pregunta puedes revisarlo.", "Puedes. Revisarlo.", "¿Puedes revisarlo?"),
        )
        for raw, output, expected in cases:
            with self.subTest(output=output):
                result, _ = self.cleanup(raw, output)
                self.assertEqual(result, expected)

    def test_v30_question_keeps_decimal_and_technical_identifier(self):
        result, _ = self.cleanup(
            "Pregunta funciona la versión 4.58.0 en proyecto room guion manager.",
            "Funciona la versión 4.58.0 en proyecto room-manager.",
        )
        self.assertEqual(result, "¿Funciona la versión 4.58.0 en proyecto room-manager?")

    def test_v31_punctuation_only_questions_pass_safety_guards(self):
        cases = (
            ("Como funciona esto.", "¿Cómo funciona esto?"),
            ("Hay alguna forma de mejorar esto.", "¿Hay alguna forma de mejorar esto?"),
            ("Puedes enviarme el archivo.", "¿Puedes enviarme el archivo?"),
            ("Ya terminé. Puedes revisarlo.", "Ya terminé. ¿Puedes revisarlo?"),
            ("No sé cómo funciona esto.", "No sé cómo funciona esto."),
            ("Quería saber si puedes enviarme el archivo.", "Quería saber si puedes enviarme el archivo."),
            ("Mañana vienes.", "Mañana vienes."),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, expected)
                self.assertEqual(result, expected)

    def test_v31_prompt_contains_direct_indirect_and_ambiguous_examples(self):
        prompt = asr_server.DEFAULT_POSTPROCESS_PROMPT
        for example in (
            "Raw: Puedes enviarme el archivo.\nClean: ¿Puedes enviarme el archivo?",
            "Raw: No sé cómo funciona esto.\nClean: No sé cómo funciona esto.",
            "Raw: Mañana vienes.\nClean: Mañana vienes.",
        ):
            self.assertIn(example, prompt)

    def test_v31_indirect_questions_reject_added_question_marks_and_word_loss(self):
        cases = (
            ("Dime dónde está el archivo.", "¿Dónde está el archivo?", "Dime dónde está el archivo."),
            ("Con este permiso puedes editar el archivo.", "Con este permiso, ¿puedes editar el archivo?", "Con este permiso, puedes editar el archivo."),
            ("Gracias a este cambio puedes enviar archivos.", "Gracias a este cambio, ¿puedes enviar archivos?", "Gracias a este cambio, puedes enviar archivos."),
            ("No sé cómo funciona. Puedes revisarlo.", "¿No sé cómo funciona? ¿Puedes revisarlo?", "No sé cómo funciona. ¿Puedes revisarlo?"),
        )
        for raw, output, expected in cases:
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, output)
                self.assertEqual(result, expected)

    def test_v31_explicit_questions_override_declarative_prefix_protection(self):
        cases = (
            ("¿Con este permiso puedes editar el archivo?", "¿Con este permiso puedes editar el archivo?"),
            ("Pregunta con este permiso puedes editar el archivo.", "Con este permiso puedes editar el archivo."),
        )
        for raw, output in cases:
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, output)
                self.assertEqual(result, "¿Con este permiso puedes editar el archivo?")

    def test_v35_explanatory_que_rejects_questions_and_accent_changes(self):
        cases = (
            ("Disculpa, que no te había leído.", "Disculpa, ¿qué no te había leído?"),
            ("Perdona que no te haya contestado.", "Perdona, ¿qué no te haya contestado?"),
            ("Que mañana no puedo.", "¿Qué mañana no puedo?"),
            ("Que no te había leído.", "¿Qué no te había leído?"),
            ("Disculpa, que no te había leído.", "Disculpa, qué no te había leído."),
            ("Disculpa, que no te había leído. Puedes revisarlo.",
             "Disculpa, ¿qué no te había leído? ¿Puedes revisarlo?"),
        )
        for raw, output in cases:
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, output)
                self.assertNotIn("¿qué", result.lower())
                self.assertNotIn("¿que", result.lower())
                self.assertIn("que", result.lower())
                if "Puedes revisarlo" in raw:
                    self.assertTrue(result.endswith("¿Puedes revisarlo?"))

    def test_v35_explanation_preserves_embedded_interrogative_accent(self):
        result, _ = self.cleanup(
            "Disculpa, que no sé qué has leído.",
            "Disculpa, ¿que no sé qué has leído?",
        )
        self.assertEqual(result, "Disculpa, que no sé qué has leído.")

    def test_v35_echo_questions_keep_unaccented_que(self):
        for raw in ("¿Que mañana no puedes?", "Pregunta que mañana no puedes."):
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, "¿Qué mañana no puedes?")
                self.assertEqual(result, "¿Que mañana no puedes?")

    def test_v35_real_questions_survive_explanatory_protection(self):
        cases = (
            ("Disculpa, qué habías leído.", "Disculpa, ¿qué habías leído?"),
            ("¿Que mañana no puedes?", "¿Que mañana no puedes?"),
            ("Pregunta que mañana no puedes.", "¿Que mañana no puedes?"),
            ("Pregunta disculpa que no te había leído.", "¿Disculpa que no te había leído?"),
            ("Qué quieres que haga.", "¿Qué quieres que haga?"),
        )
        for raw, output in cases:
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, output)
                self.assertEqual(result, output)

    def test_v36_haber_conjugations_survive_cleanup(self):
        cases = (
            ("Disculpa, que no te había leído.", "Disculpa, ¿qué no te habías leído?"),
            ("Voy a abrirte la puerta. Disculpa que no te había leído.",
             "Voy a abrirte la puerta. Disculpa que no te hubieras leído."),
            ("No te había leído. Puedes revisarlo.", "No te habías leído. ¿Puedes revisarlo?"),
            ("Espero que haya llegado.", "Espero que ha llegado."),
            ("Ya hemos terminado.", "Ya han terminado."),
            ("¿Habías leído esto?", "¿Había leído esto?"),
            ("Habíamos terminado y habías avisado.", "Habían terminado y había avisado."),
        )
        for raw, output in cases:
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, output)
                expected = raw.replace("Puedes revisarlo.", "¿Puedes revisarlo?")
                self.assertEqual(result, expected)

    def test_v36_haber_protection_accepts_formatting_and_missing_auxiliary(self):
        cases = (
            ("cuando habias llegado", "¿Cuándo habías llegado?"),
            ("me dicho que viene", "Me ha dicho que viene."),
            ("Había 25 personas.", "Había 25 personas."),
        )
        for raw, output in cases:
            with self.subTest(raw=raw):
                result, _ = self.cleanup(raw, output)
                self.assertEqual(result, output)

    def test_v36_unaligned_haber_changes_fall_back_without_guessing(self):
        raw = "Disculpa que no te había leído antes de abrir la puerta."
        result, _ = self.cleanup(raw, "Perdona que no te hubieras leído antes de abrir la puerta.")
        self.assertEqual(result, raw)


if __name__ == "__main__":
    unittest.main()
