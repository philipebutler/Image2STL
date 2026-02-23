"""Tests for core/project.py - Project data model."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from core.project import Project, ProjectSettings, SUPPORTED_IMAGE_EXTENSIONS, PROJECT_FILE_NAME


class TestProjectCreate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_new_sets_fields(self):
        p = Project.create_new("TestScan", self.base)
        self.assertEqual(p.name, "TestScan")
        self.assertIsNotNone(p.project_id)
        self.assertIsNotNone(p.created)
        self.assertIsNotNone(p.last_modified)
        self.assertEqual(p.images, [])
        self.assertIsNone(p.model_path)

    def test_create_new_creates_directories(self):
        p = Project.create_new("DirTest", self.base)
        self.assertTrue(p.project_path.is_dir())
        self.assertTrue(p.images_dir.is_dir())
        self.assertTrue(p.models_dir.is_dir())
        self.assertTrue(p.preview_dir.is_dir())

    def test_create_new_saves_project_json(self):
        p = Project.create_new("SaveTest", self.base)
        project_file = p.project_path / PROJECT_FILE_NAME
        self.assertTrue(project_file.exists())
        data = json.loads(project_file.read_text(encoding="utf-8"))
        self.assertEqual(data["name"], "SaveTest")


class TestProjectSaveLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip(self):
        original = Project.create_new("RoundTrip", self.base)
        original.settings.scale_mm = 200.0
        original.save()

        loaded = Project.load(original.project_path)
        self.assertEqual(loaded.name, "RoundTrip")
        self.assertEqual(loaded.project_id, original.project_id)
        self.assertAlmostEqual(loaded.settings.scale_mm, 200.0)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            Project.load(self.base / "nonexistent")

    def test_from_dict_roundtrip(self):
        original = Project.create_new("DictTest", self.base)
        original.settings.reconstruction_mode = "cloud"
        data = original.to_dict()
        restored = Project.from_dict(data)
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.settings.reconstruction_mode, "cloud")


class TestProjectAddImage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = Path(self.tmp)
        self.project = Project.create_new("ImageTest", self.base)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_image(self, name: str) -> Path:
        img = self.base / name
        img.write_bytes(b"fake_image_data")
        return img

    def test_add_jpg_image(self):
        img = self._make_image("photo.jpg")
        rel = self.project.add_image(img)
        self.assertIn(rel, self.project.images)
        self.assertTrue((self.project.project_path / rel).exists())

    def test_add_png_image(self):
        img = self._make_image("photo.png")
        rel = self.project.add_image(img)
        self.assertTrue(rel.endswith(".png"))

    def test_add_heic_image(self):
        img = self._make_image("photo.heic")
        rel = self.project.add_image(img)
        self.assertTrue(rel.endswith(".heic"))

    def test_add_heif_image(self):
        img = self._make_image("photo.heif")
        rel = self.project.add_image(img)
        self.assertTrue(rel.endswith(".heif"))

    def test_add_webp_image(self):
        img = self._make_image("photo.webp")
        rel = self.project.add_image(img)
        self.assertTrue(rel.endswith(".webp"))

    def test_add_avif_image(self):
        img = self._make_image("photo.avif")
        rel = self.project.add_image(img)
        self.assertTrue(rel.endswith(".avif"))

    def test_add_unsupported_format_raises(self):
        img = self._make_image("photo.bmp")
        with self.assertRaises(ValueError):
            self.project.add_image(img)

    def test_add_multiple_images(self):
        for name in ("a.jpg", "b.png", "c.heic"):
            self.project.add_image(self._make_image(name))
        self.assertEqual(len(self.project.images), 3)

    def test_duplicate_name_handled(self):
        img = self._make_image("photo.jpg")
        rel1 = self.project.add_image(img)

        img2 = self.base / "photo.jpg"  # same name, different source
        img2.write_bytes(b"other_data")
        rel2 = self.project.add_image(img2)

        self.assertNotEqual(rel1, rel2)
        self.assertEqual(len(self.project.images), 2)


class TestProjectRemoveImage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = Path(self.tmp)
        self.project = Project.create_new("RemoveTest", self.base)
        img = self.base / "photo.jpg"
        img.write_bytes(b"fake_data")
        self.rel_path = self.project.add_image(img)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_remove_image(self):
        self.project.remove_image(self.rel_path)
        self.assertNotIn(self.rel_path, self.project.images)

    def test_remove_image_deletes_file(self):
        full = self.project.project_path / self.rel_path
        self.project.remove_image(self.rel_path)
        self.assertFalse(full.exists())

    def test_remove_nonexistent_is_safe(self):
        self.project.remove_image("images/ghost.jpg")  # should not raise
        self.assertEqual(len(self.project.images), 1)


class TestProjectSettings(unittest.TestCase):
    def test_defaults(self):
        s = ProjectSettings()
        self.assertEqual(s.reconstruction_mode, "local")
        self.assertAlmostEqual(s.scale_mm, 150.0)
        self.assertEqual(s.scale_axis, "longest")
        self.assertEqual(s.mesh_quality, "medium")


class TestSupportedExtensions(unittest.TestCase):
    def test_contains_expected_formats(self):
        for ext in (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".avif"):
            self.assertIn(ext, SUPPORTED_IMAGE_EXTENSIONS)

    def test_bmp_not_supported(self):
        self.assertNotIn(".bmp", SUPPORTED_IMAGE_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
