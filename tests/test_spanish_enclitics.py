import unittest
from unittest.mock import patch

import asr_server as server


def response(text):
    return {"choices": [{"message": {"content": text}}]}


class SpanishEncliticsTest(unittest.TestCase):
    def cleanup(self, raw, edited=None):
        with (
            patch.object(server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(server, "ASR_POSTPROCESS_VALIDATE_EDITS", True),
            patch.object(server, "_llm_chat", return_value=response(edited or raw)),
        ):
            return server._postprocess_transcript(raw)

    def test_v54_repairs_standalone_commands_even_when_model_copies_source(self):
        for raw, expected in [
            ("Perfecto, implementa lo.", "Perfecto, impleméntalo."),
            ("Implementa lo", "Impleméntalo"),
            ("Vale, revisa lo.", "Vale, revísalo."),
            ("Por favor, haz lo.", "Por favor, hazlo."),
            ("Prueba lo. Después seguimos.", "Pruébalo. Después seguimos."),
            ("Gracias. Implementa lo.", "Gracias. Impleméntalo."),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(self.cleanup(raw), expected)

    def test_v54_preserves_relative_clauses_articles_and_ambiguous_context(self):
        for raw in [
            "Implementa lo que hemos hablado.",
            "Implementa lo de ayer.",
            "Haz lo posible.",
            "Revisa lo importante antes de continuar.",
            "Implementa los cambios.",
            "Ella implementa lo.",
            "No implementa lo.",
            "La palabra es implementa lo.",
            'Escribe "implementa lo".',
            "Implementa lo mismo que ayer.",
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(self.cleanup(raw), raw)

    def test_v54_accepts_joined_output_without_losing_pronoun_protection(self):
        self.assertEqual(self.cleanup("Perfecto, implementa lo.", "Perfecto, impleméntalo."),
                         "Perfecto, impleméntalo.")
        raw = "Implementa lo que hemos hablado."
        self.assertEqual(self.cleanup(raw, "Impleméntalo que hemos hablado."), raw)

    def test_v54_disabled_or_unavailable_cleanup_retains_raw_contract(self):
        raw = "Perfecto, implementa lo."
        with patch.object(server, "ASR_POSTPROCESS_ENABLED", False):
            self.assertEqual(server._postprocess_transcript(raw), raw)
        with (
            patch.object(server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(server, "_llm_chat", side_effect=RuntimeError("unavailable")),
        ):
            self.assertEqual(server._postprocess_transcript(raw), raw)
