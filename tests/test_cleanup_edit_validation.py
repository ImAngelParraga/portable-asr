import json
import unittest
from unittest.mock import patch

import asr_server as server


def response(text):
    return {"choices": [{"message": {"content": text}}]}


class CleanupEditValidationTest(unittest.TestCase):
    def run_cleanup(self, raw, edited, verdict="A"):
        with (
            patch.object(server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(server, "ASR_POSTPROCESS_VALIDATE_EDITS", True, create=True),
            patch.object(server, "_llm_chat", side_effect=[response(edited), verdict if isinstance(verdict, Exception) else response(verdict)]) as chat,
        ):
            result = server._postprocess_transcript(raw, fail_closed=True)
        return result, chat

    def test_v48_rejects_small_tense_and_person_changes_in_long_sentence(self):
        raw = "Otros no leemos tus mensajes, pero tú tampoco lees todo lo que mando."
        edited = "Otros no leen tus mensajes, pero tú tampoco lees todo lo que mandé."
        final, chat = self.run_cleanup(raw, edited, "B")
        self.assertEqual(final, raw)
        self.assertEqual(chat.call_count, 1)

    def test_v48_audits_accent_only_tense_change(self):
        raw = "Yo mando los documentos cuando termina la reunión."
        final, chat = self.run_cleanup(raw, raw.replace("mando", "mandó"), "B")
        self.assertEqual(final, raw)
        self.assertEqual(chat.call_count, 1)

    def test_v48_allows_audited_spelling_and_missing_function_words(self):
        pairs = [
            ("ramon me dicho que no le gusta tu actitud", "Ramón me ha dicho que no le gusta tu actitud."),
            ("Voy hablar con Ana después de la reunión.", "Voy a hablar con Ana después de la reunión."),
            ("La respuesta del serividor tarda mucho cuando hay varias conexiones.",
             "La respuesta del servidor tarda mucho cuando hay varias conexiones."),
        ]
        for raw, edited in pairs:
            with self.subTest(raw=raw):
                final, chat = self.run_cleanup(raw, edited)
                self.assertEqual(final, edited)
                self.assertEqual(chat.call_count, 1 if raw.startswith("Voy hablar") else 2)

    def test_v48_bounded_ir_and_tener_function_insertions_need_no_audit(self):
        for raw, edited in [
            ("Vamos revisar los documentos antes de continuar.", "Vamos a revisar los documentos antes de continuar."),
            ("Tienen comprobar los cambios antes de continuar.", "Tienen que comprobar los cambios antes de continuar."),
        ]:
            with self.subTest(raw=raw):
                final, chat = self.run_cleanup(raw, edited, "B")
                self.assertEqual(final, edited)
                self.assertEqual(chat.call_count, 1)

    def test_v48_punctuation_capitalization_and_numeric_format_skip_audit(self):
        pairs = [
            ("quiero que funcione", "Quiero que funcione."),
            ("La versión es cuatro punto cincuenta y ocho punto cero.", "La versión es 4.58.0."),
        ]
        for raw, edited in pairs:
            with self.subTest(raw=raw):
                final, chat = self.run_cleanup(raw, edited)
                self.assertEqual(final, edited)
                self.assertEqual(chat.call_count, 1)

    def test_v48_invalid_or_unavailable_audit_never_releases_rewrite(self):
        raw = "Guardo el informe en la carpeta que usamos para la reunión."
        for verdict in ("", "Sí", "A because it looks correct", RuntimeError("unavailable")):
            with self.subTest(verdict=str(verdict)):
                final, chat = self.run_cleanup(raw, raw.replace("informe", "archivo"), verdict)
                self.assertEqual(final, raw)
                self.assertEqual(chat.call_count, 2)

    def test_v48_preserves_irregular_forms_and_multiple_conjugation_families(self):
        for before, after in [("puedo", "podría"), ("voy", "fui"), ("somos", "eran"),
                              ("tenemos", "tuvimos"), ("envío", "envió"), ("hablo", "habló")]:
            raw = f"Yo {before} cuando termina la reunión."
            with self.subTest(before=before, after=after):
                final, chat = self.run_cleanup(raw, raw.replace(before, after))
                self.assertEqual(final, raw)
                self.assertEqual(chat.call_count, 1)

    def test_v48_keeps_repeated_negation_without_asking_model(self):
        raw = "No, no he dicho que el equipo deba reiniciarse ahora mismo."
        final, chat = self.run_cleanup(raw, raw.replace("No, no", "No"))
        self.assertEqual(final, raw)
        self.assertEqual(chat.call_count, 1)

    def test_v48_missing_lexicon_retains_source(self):
        raw = "Quiero que el equipo funcione correctamente."
        with patch.object(server, "_spanish_verb_lexicon", return_value=None):
            final, chat = self.run_cleanup(raw, raw.replace("equipo", "servidor"))
        self.assertEqual(final, raw)
        self.assertEqual(chat.call_count, 1)

    def test_v48_preserves_ambiguous_valid_forms_but_allows_missing_accent_spelling(self):
        raw = "Mañana revisare la configuración del teclado."
        final, chat = self.run_cleanup(raw, raw.replace("revisare", "revisaré"))
        self.assertEqual(final, raw)
        self.assertEqual(chat.call_count, 1)
        raw = "Si hubieramos tenido tiempo, lo habríamos probado."
        edited = raw.replace("hubieramos", "hubiéramos")
        final, chat = self.run_cleanup(raw, edited)
        self.assertEqual(final, edited)
        self.assertEqual(chat.call_count, 1)

    def test_v48_rejection_still_applies_explicit_question_control(self):
        final, _ = self.run_cleanup("Pregunta puedes revisar los documentos que mando.",
                                    "¿Puedes revisar los documentos que mandé?", "B")
        self.assertIn("mando", final)
        self.assertTrue(final.startswith("¿"))
        self.assertTrue(final.endswith("?"))

    def test_v48_disabled_validation_preserves_legacy_single_call(self):
        raw = "Yo mando el informe cuando termina la reunión."
        edited = raw.replace("mando", "mandé")
        with (
            patch.object(server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(server, "ASR_POSTPROCESS_VALIDATE_EDITS", False, create=True),
            patch.object(server, "_llm_chat", return_value=response(edited)) as chat,
        ):
            self.assertEqual(server._postprocess_transcript(raw), edited)
        self.assertEqual(chat.call_count, 1)


if __name__ == "__main__":
    unittest.main()
