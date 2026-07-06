import unittest

from asr_server import _apply_explicit_paragraph_breaks, _remove_quoted_and_empty_wrapper_lines


class OutputCleanupTest(unittest.TestCase):
    def test_preserves_paragraph_breaks(self):
        text = "Primer tema.\n\nSegundo tema."
        self.assertEqual(_remove_quoted_and_empty_wrapper_lines(text), text)

    def test_collapses_excess_blank_lines(self):
        self.assertEqual(
            _remove_quoted_and_empty_wrapper_lines("Primer tema.\n\n\n\nSegundo tema."),
            "Primer tema.\n\nSegundo tema.",
        )

    def test_removes_quoted_wrapper_lines(self):
        self.assertEqual(
            _remove_quoted_and_empty_wrapper_lines("> wrapper\nPrimer tema.\n\nSegundo tema."),
            "Primer tema.\n\nSegundo tema.",
        )

    def test_adds_paragraph_before_explicit_topic_transition(self):
        self.assertEqual(
            _apply_explicit_paragraph_breaks(
                "Estoy revisando el servidor. Ahora cambio completamente de tema y hablo del partido."
            ),
            "Estoy revisando el servidor.\n\nAhora cambio completamente de tema y hablo del partido.",
        )

    def test_adds_paragraph_before_dicho_esto(self):
        self.assertEqual(
            _apply_explicit_paragraph_breaks("Estoy probando esto. Dicho esto, hablemos del partido."),
            "Estoy probando esto.\n\nDicho esto, hablemos del partido.",
        )

    def test_adds_paragraph_before_por_cierto(self):
        self.assertEqual(
            _apply_explicit_paragraph_breaks("¡Vamos, Japón! Ah, por cierto: llamé a mi mejor amigo."),
            "¡Vamos, Japón!\n\nAh, por cierto: llamé a mi mejor amigo.",
        )


if __name__ == "__main__":
    unittest.main()
