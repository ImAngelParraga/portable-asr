import unittest
from unittest.mock import patch

import asr_server
from asr_server import _postprocess_transcript, _prepare_formatted_list_transcript


class FormattedListTest(unittest.TestCase):
    def _chat_result(self, content):
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {},
        }

    def _postprocess_with_model_output(self, raw, model_output):
        captured = {}

        def fake_chat(messages, **kwargs):
            captured["user_prompt"] = messages[-1]["content"]
            return self._chat_result(model_output)

        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", side_effect=fake_chat),
        ):
            result = _postprocess_transcript(raw)
        return result, captured["user_prompt"]

    def test_spanish_bullet_list_uses_implicit_intro_colon_and_comma_items(self):
        raw = "hoy tengo que comprar lista de items plátanos, tomates"
        self.assertEqual(
            _prepare_formatted_list_transcript(raw),
            "hoy tengo que comprar:\n\n- plátanos\n- tomates",
        )

    def test_english_bullet_list_uses_shared_new_item_separator(self):
        raw = "today I need to buy item list bananas new item tomatoes"
        self.assertEqual(
            _prepare_formatted_list_transcript(raw),
            "today I need to buy:\n\n- bananas\n- tomatoes",
        )

    def test_spanish_numbered_list_uses_shared_new_item_separator(self):
        raw = "para desplegar lista numerada ejecutar las pruebas nuevo ítem copiar el archivo"
        self.assertEqual(
            _prepare_formatted_list_transcript(raw),
            "para desplegar:\n\n1. ejecutar las pruebas\n2. copiar el archivo",
        )

    def test_english_numbered_list_uses_shared_new_item_separator(self):
        raw = "deployment steps numbered list run tests new item copy the file"
        self.assertEqual(
            _prepare_formatted_list_transcript(raw),
            "deployment steps:\n\n1. run tests\n2. copy the file",
        )

    def test_list_without_intro_has_no_invented_title(self):
        raw = "lista de ítems pan nuevo item leche"
        self.assertEqual(
            _prepare_formatted_list_transcript(raw),
            "- pan\n- leche",
        )

    def test_end_of_list_preserves_following_prose(self):
        raw = (
            "hoy debo comprar lista de ítems pan nuevo ítem leche "
            "fin de lista mañana cocinaré"
        )
        self.assertEqual(
            _prepare_formatted_list_transcript(raw),
            "hoy debo comprar:\n\n- pan\n- leche\n\nmañana cocinaré",
        )

    def test_ordinary_spanish_list_phrase_remains_prose(self):
        raw = "estoy revisando una lista de ítems defectuosos"
        self.assertEqual(_prepare_formatted_list_transcript(raw), raw)

    def test_ordinary_english_list_phrase_remains_prose(self):
        raw = "we discussed an item list, but did not create one"
        self.assertEqual(_prepare_formatted_list_transcript(raw), raw)

    def test_preformatted_list_reaches_model_and_survives_safety_pipeline(self):
        raw = "hoy tengo que comprar lista de items plátanos nuevo ítem tomates"
        corrected = "Hoy tengo que comprar:\n\n- Plátanos\n- Tomates"
        result, user_prompt = self._postprocess_with_model_output(raw, corrected)
        self.assertIn(
            "Raw transcript:\nhoy tengo que comprar:\n\n- plátanos\n- tomates",
            user_prompt,
        )
        self.assertEqual(result, corrected)

    def test_numbered_list_survives_safety_pipeline(self):
        raw = "deployment steps numbered list run tests new item copy files"
        corrected = "Deployment steps:\n\n1. Run tests\n2. Copy files"
        result, _ = self._postprocess_with_model_output(raw, corrected)
        self.assertEqual(result, corrected)

    def test_deleted_item_falls_back_to_preformatted_source(self):
        raw = "comprar lista de ítems pan nuevo ítem leche nuevo ítem tomates"
        corrected = "Comprar:\n\n- Pan\n- Tomates"
        result, _ = self._postprocess_with_model_output(raw, corrected)
        self.assertEqual(
            result,
            "comprar:\n\n- pan\n- leche\n- tomates",
        )

    def test_reordered_items_fall_back_to_preformatted_source(self):
        raw = "comprar item list bread new item milk new item tomatoes"
        corrected = "Comprar:\n\n- Tomatoes\n- Milk\n- Bread"
        result, _ = self._postprocess_with_model_output(raw, corrected)
        self.assertEqual(
            result,
            "comprar:\n\n- bread\n- milk\n- tomatoes",
        )

    def test_invented_item_content_falls_back_to_preformatted_source(self):
        raw = "comprar lista de ítems pan nuevo ítem leche"
        corrected = "Comprar:\n\n- Pan integral ecológico\n- Leche"
        result, _ = self._postprocess_with_model_output(raw, corrected)
        self.assertEqual(result, "comprar:\n\n- pan\n- leche")

    def test_prompt_documents_bilingual_list_controls(self):
        prompt = asr_server.DEFAULT_POSTPROCESS_PROMPT
        for control in (
            "lista de ítems",
            "item list",
            "lista numerada",
            "numbered list",
            "nuevo ítem",
            "new item",
            "fin de lista",
            "end of list",
        ):
            with self.subTest(control=control):
                self.assertIn(control, prompt)


if __name__ == "__main__":
    unittest.main()
