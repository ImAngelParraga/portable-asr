import unittest

from asr_server import (
    _apply_explicit_paragraph_breaks,
    _remove_quoted_and_empty_wrapper_lines,
    _repair_technical_literal_format,
)


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

    def test_reconstructs_guion_bajo_inside_technical_literal(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "b2b guion bajo round guion bajo history",
                "b2b guion bajo round guion bajo history",
            ),
            "b2b_round_history",
        )

    def test_reconstructs_barra_baja_inside_technical_literal(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "b2b barra baja round barra baja history",
                "b2b barra baja round barra baja history",
            ),
            "b2b_round_history",
        )

    def test_underscore_hint_fixes_hyphenated_capitalized_identifier(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "b2b guion bajo round guion bajo history",
                "B2b-Round-History",
            ),
            "b2b_round_history",
        )

    def test_underscore_hint_supports_accented_identifier_words(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "contraseña guion bajo temporal",
                "Contraseña-Temporal",
            ),
            "contraseña_temporal",
        )

    def test_reconstructs_unmodified_accented_spoken_literal(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "contraseña guion bajo temporal",
                "contraseña guion bajo temporal",
            ),
            "contraseña_temporal",
        )

    def test_collapses_joined_spoken_separator_words(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "contraseña guion bajo temporal",
                "contraseña_guion_bajo_temporal",
            ),
            "contraseña_temporal",
        )

    def test_reconstructs_spoken_literal_after_technical_cue(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "la variable contraseña guion bajo temporal",
                "La variable contraseña guion bajo temporal.",
            ),
            "La variable contraseña_temporal.",
        )

    def test_underscore_hint_only_repairs_matching_sequence(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "b2b guion bajo history y state of the art",
                "B2b-History y state-of-the-art.",
            ),
            "b2b_history y state-of-the-art.",
        )

    def test_hyphen_hint_repairs_wrong_underscore(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "frontend guion backend",
                "Frontend_Backend",
            ),
            "frontend-backend",
        )

    def test_normal_sentence_keeps_barra_baja_words(self):
        self.assertEqual(
            _repair_technical_literal_format(
                "la barra baja está mal",
                "La barra baja está mal.",
            ),
            "La barra baja está mal.",
        )

    def test_v11_bilingual_context_cues_reconstruct_single_separator(self):
        cases = (
            ("variable", "hyphen"),
            ("field", "hyphen"),
            ("identifier", "hyphen"),
            ("file", "hyphen"),
            ("filename", "hyphen"),
            ("name", "hyphen"),
            ("token", "hyphen"),
            ("key", "hyphen"),
            ("column", "hyphen"),
            ("table", "hyphen"),
            ("class", "hyphen"),
            ("function", "hyphen"),
            ("method", "hyphen"),
            ("endpoint", "hyphen"),
            ("folder", "hyphen"),
            ("directory", "hyphen"),
            ("path", "hyphen"),
            ("document", "hyphen"),
            ("campo", "guion"),
            ("identificador", "guion"),
            ("archivo", "guion"),
            ("fichero", "guion"),
            ("nombre", "guion"),
            ("clave", "guion"),
            ("columna", "guion"),
            ("tabla", "guion"),
            ("clase", "guion"),
            ("función", "guion"),
            ("método", "guion"),
            ("carpeta", "guion"),
            ("directorio", "guion"),
            ("ruta", "guion"),
            ("documento", "guion"),
        )
        for cue, spoken_separator in cases:
            with self.subTest(cue=cue):
                self.assertEqual(
                    _repair_technical_literal_format(
                        f"{cue} Alpha {spoken_separator} Beta",
                        f"{cue} Alpha Beta",
                    ),
                    f"{cue} alpha-beta",
                )


if __name__ == "__main__":
    unittest.main()
