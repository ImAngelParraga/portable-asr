import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import asr_server


class CleanupPromptConfigTest(unittest.TestCase):
    def test_v51_unset_keeps_builtin(self):
        with patch.object(asr_server, "ASR_POSTPROCESS_PROMPT_FILE", ""):
            self.assertEqual(asr_server._load_postprocess_prompt("builtin"), "builtin")

    def test_v51_file_loads_on_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("  Preserve words and accents: café.\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-c", "import asr_server; print(asr_server.DEFAULT_POSTPROCESS_PROMPT)"],
                env={**os.environ, "ASR_POSTPROCESS_PROMPT_FILE": str(path)},
                cwd=Path(asr_server.__file__).parent,
                capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "Preserve words and accents: café.")

    def test_v51_invalid_files_fail_without_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in (("blank", b" \n"), ("invalid", bytes([255]))):
                (root / name).write_bytes(content)
            for name in ("missing", "blank", "invalid", "."):
                with self.subTest(name=name), patch.object(
                    asr_server, "ASR_POSTPROCESS_PROMPT_FILE", str(root / name)
                ):
                    with self.assertRaisesRegex(ValueError, "ASR_POSTPROCESS_PROMPT_FILE"):
                        asr_server._load_postprocess_prompt("builtin")

    def test_v51_request_override_and_default_reach_cleanup_model(self):
        for instruction, expected in ((None, "Configured instructions"), ("Request instructions", "Request instructions")):
            with (
                self.subTest(instruction=instruction),
                patch.object(asr_server, "ASR_POSTPROCESS_ENABLED", True),
                patch.object(asr_server, "DEFAULT_POSTPROCESS_PROMPT", "Configured instructions"),
                patch.object(asr_server, "_llm_chat", return_value={
                    "choices": [{"message": {"content": "Hello world."}}]
                }) as chat,
            ):
                self.assertEqual(asr_server._postprocess_transcript("hello world", instruction), "Hello world.")
                self.assertEqual(chat.call_args.args[0][0]["content"], expected)


if __name__ == "__main__":
    unittest.main()
