import unittest
from pathlib import Path


class PackagingFilesTestCase(unittest.TestCase):
    def test_pyinstaller_spec_includes_ui_data_and_hidden_imports(self):
        spec = Path("medical_qa_app.spec").read_text(encoding="utf-8")

        self.assertIn("ui_preview.html", spec)
        self.assertIn("Data_数据", spec)
        self.assertIn("MedicalQA", spec)
        self.assertIn("importlib.resources", spec)
        self.assertIn("sklearn.feature_extraction.text", spec)
        self.assertIn("scipy.sparse", spec)

    def test_build_script_invokes_pyinstaller_spec(self):
        script = Path("build_exe.ps1").read_text(encoding="utf-8")

        self.assertIn("medical_qa_app.spec", script)
        self.assertIn("PyInstaller", script)
        self.assertIn("dist", script)
        self.assertIn("$LASTEXITCODE", script)


if __name__ == "__main__":
    unittest.main()
