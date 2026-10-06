import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import asr_server


class QuantitySymbolsTest(unittest.TestCase):
    def clean(self, raw, output):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", return_value={
                "choices": [{"message": {"content": output}}], "usage": {},
            }),
        ):
            return asr_server._postprocess_transcript(raw)

    def test_v32_spoken_quantities_convert_even_when_model_keeps_words(self):
        for raw, expected in (
            ("doscientos euros", "200 €"),
            ("cinco dólares", "$5"),
            ("veintidós euros", "22 €"),
            ("one hundred and twenty-five dollars", "$125"),
            ("twenty five euros", "25 €"),
            ("diez por ciento", "10%"),
            ("ten percent", "10%"),
            ("un euro", "1 €"),
            ("Cuesta 1.250,50 euros y 12.50 dólares.", "Cuesta 1.250,50 € y $12.50."),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(self.clean(raw, raw), expected)
                self.assertEqual(self.clean(raw, expected), expected)

    def test_v32_equivalent_prefix_and_suffix_currency_forms_are_accepted(self):
        self.assertEqual(self.clean("cuesta cinco dólares", "Cuesta 5 $."), "Cuesta $5.")
        self.assertEqual(self.clean("cuesta doscientos euros", "Cuesta €200."), "Cuesta 200 €.")

    def test_v32_changed_currency_amount_and_words_are_rejected(self):
        raw = "paga doscientos euros mañana"
        for output in ("Paga $200 mañana.", "Paga 201 € mañana.",
                       "Paga 200 mañana.", "Paga 200 € hoy.", "Paga 200% mañana."):
            with self.subTest(output=output):
                self.assertEqual(self.clean(raw, output), "paga 200 € mañana")

    def test_v32_ordinary_currency_mentions_and_unsupported_numbers_remain_literal(self):
        for raw in ("el euro sube frente al dólar", "euros y dólares", "euro dólar",
                    "mil doscientos euros", "one thousand five dollars"):
            with self.subTest(raw=raw):
                self.assertEqual(self.clean(raw, raw), raw)

    def test_v32_quantities_inside_lists_and_questions(self):
        self.assertEqual(self.clean("pregunta cuesta veinte euros", "Cuesta veinte euros."),
                         "¿Cuesta 20 €?")
        self.assertEqual(self.clean("comprar lista de ítems pan cinco euros nuevo ítem leche dos dólares",
                                   "Comprar:\n\n- Pan 5 €\n- Leche $2"),
                         "Comprar:\n\n- Pan 5 €\n- Leche $2")

    def test_v32_disabled_and_unavailable_cleanup_preserve_raw_contract(self):
        raw = "doscientos euros"
        with patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", False):
            self.assertEqual(asr_server._postprocess_transcript(raw), raw)
        with (patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
              patch.object(asr_server, "_llm_chat", side_effect=RuntimeError("unavailable"))):
            self.assertEqual(asr_server._postprocess_transcript(raw), raw)

    def test_v32_audio_endpoint_formats_both_transcription_engines(self):
        with (
            patch.multiple(asr_server, ASR_BEARER_TOKEN="test-token",
                           ASR_POSTPROCESS_ENABLED=True, ASR_QWEN_ENABLED=True,
                           ASR_STOP_LLM_FOR_TRANSCRIPTION=False, ASR_WHISPER_KEEP_WARM=True,
                           ASR_AUDIO_HISTORY_ENABLED=False),
            patch.object(asr_server, "_run_transcription", return_value=(["doscientos euros"], {})),
            patch.object(asr_server, "_run_qwen_transcription", return_value=(["doscientos euros"], {})),
            patch.object(asr_server, "_llm_chat", return_value={
                "choices": [{"message": {"content": "Doscientos euros."}}], "usage": {},
            }),
        ):
            client = TestClient(asr_server.app)
            for model in ("whisper-1", asr_server.ASR_QWEN_MODEL_ID):
                with self.subTest(model=model):
                    response = client.post(
                        "/v1/audio/transcriptions", headers={"Authorization": "Bearer test-token"},
                        data={"model": model, "language": "es"},
                        files={"file": ("input.wav", b"test-audio", "audio/wav")},
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["text"], "200 €.")


if __name__ == "__main__":
    unittest.main()
