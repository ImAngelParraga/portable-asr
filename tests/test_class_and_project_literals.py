import unittest
from unittest.mock import patch

import asr_server


class ClassAndProjectLiteralsTest(unittest.TestCase):
    def clean(self, raw, model_output):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", return_value={
                "choices": [{"message": {"content": model_output}}],
            }),
        ):
            return asr_server._postprocess_transcript(raw)

    def test_v28_project_separators_and_cue_case(self):
        cases = (
            ("changes in project demo hyphen tools", "Changes in Project Demo hyphen tools.", "Changes in project demo-tools."),
            ("changes in project demo hyphen tools", "Changes in Project Demo Tools.", "Changes in project demo-tools."),
            ("changes in project demo hyphen tools", "Changes in project demo-tools.", "Changes in project demo-tools."),
            ("cambios en proyecto demo guion bajo tools", "Cambios en Proyecto Demo Tools.", "Cambios en proyecto demo_tools."),
        )
        for raw, output, expected in cases:
            with self.subTest(raw=raw, output=output):
                self.assertEqual(self.clean(raw, output), expected)

    def test_v29_class_role_names_survive_unchanged_model_output(self):
        for cue in ("class", "clase"):
            for words, identifier in (("task controller", "TaskController"), ("game manager", "GameManager"), ("audio session controller", "AudioSessionController")):
                raw = f"{cue} {words}"
                with self.subTest(raw=raw):
                    self.assertEqual(self.clean(raw, raw), f"{cue} {identifier}")

    def test_v29_combined_class_and_project_sentence(self):
        raw = "Please review changes in class task controller in project demo hyphen tools."
        expected = "Please review changes in class TaskController in project demo-tools."
        for output in (raw, raw.replace("class task controller", "Class Task Controller").replace("project demo", "Project Demo"), expected):
            with self.subTest(output=output):
                self.assertEqual(self.clean(raw, output), expected)

    def test_v29_model_code_wrappers_only_removed_from_recognized_names(self):
        raw = "changes in class task controller in project demo hyphen tools; keep `custom`"
        self.assertEqual(
            self.clean(raw, "Changes in Class `TaskController` in Project `demo-tools`; keep `custom`"),
            "Changes in class TaskController in project demo-tools; keep `custom`",
        )
        self.assertEqual(self.clean("class task controller", "Class `Task Controller`"), "Class TaskController")

    def test_v29_prompt_receives_prepared_class_name(self):
        with (
            patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
            patch.object(asr_server, "_llm_chat", return_value={"choices": [{"message": {"content": "class TaskController"}}]}) as chat,
        ):
            asr_server._postprocess_transcript("class task controller")
            self.assertIn("class TaskController", chat.call_args.args[0][1]["content"])

    def test_v29_preserves_separators_ambiguity_and_ordinary_prose(self):
        unchanged = (
            "the operator controller changed",
            "class task controller changes are useful",
            "the class of the controller is unknown",
            "class first and second controller",
            "class task. Controller is running.",
            "class task controller hyphen factory",
            "class task controller underscore factory",
            "class task controller slash factory",
            "class TaskController",
            "class task service",
            "the project manager discussed the hyphen yesterday",
        )
        for raw in unchanged:
            with self.subTest(raw=raw):
                self.assertEqual(self.clean(raw, raw), raw)

    def test_v29_changed_class_name_rejected(self):
        self.assertEqual(self.clean("class task controller", "class UserController"), "class TaskController")
        raw = "Please review changes in class task controller in this release."
        self.assertEqual(
            self.clean(raw, raw.replace("task controller", "UserController")),
            raw.replace("task controller", "TaskController"),
        )

    def test_v29_preserves_acronyms_and_sentence_start_case(self):
        self.assertEqual(
            self.clean("class XML parser controller in this release", "Class XML Parser Controller in this release."),
            "Class XMLParserController in this release.",
        )

    def test_v29_repeated_class_names_cannot_disappear(self):
        raw = "class task controller and class task controller"
        self.assertEqual(
            self.clean(raw, "class TaskController and class UserController"),
            "class TaskController and class TaskController",
        )


if __name__ == "__main__":
    unittest.main()
