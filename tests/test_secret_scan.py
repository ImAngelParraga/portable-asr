import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "secret-scan.sh"


class SecretScanTest(unittest.TestCase):
    def scan(self, content=None, initialize=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if initialize:
                subprocess.run(["git", "init", "-q", directory], check=True)
            if content is not None:
                (root / "sample file.txt").write_text(content)
                subprocess.run(["git", "-C", directory, "add", "sample file.txt"], check=True)
            return subprocess.run(["bash", str(SCRIPT)], cwd=root, capture_output=True, text=True)

    def test_v50_missing_repository_fails(self):
        self.assertNotEqual(self.scan(initialize=False).returncode, 0)

    def test_v50_empty_repository_passes(self):
        self.assertEqual(self.scan().returncode, 0)

    def test_v50_placeholder_passes(self):
        self.assertEqual(self.scan("ASR_BEARER_TOKEN=change-me\n").returncode, 0)

    def test_v50_code_assignment_is_not_a_credential(self):
        result = self.scan("candidate.asr.bearer_token = previous.asr.bearer_token.clone();")
        self.assertEqual(result.returncode, 0)

    def test_v50_private_path_reports_only_location(self):
        private_path = "/home/" + "example/private-file"
        result = self.scan(private_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sample file.txt:1:", result.stdout)
        self.assertNotIn(private_path, result.stdout + result.stderr)

    def test_v50_plain_and_quoted_tokens_fail_without_disclosure(self):
        token = "synthetic" * 5
        for quote in ("", chr(34), chr(39)):
            with self.subTest(quote=quote):
                result = self.scan("ASR_BEARER_TOKEN=" + quote + token + quote)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn(token, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
