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


if __name__ == "__main__":
    unittest.main()
