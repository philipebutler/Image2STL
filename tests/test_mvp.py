import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from unittest.mock import patch

from image2stl import cli
from image2stl.engine import calculate_scale_factor, parse_json_line, process_command
from image2stl.project import create_project, load_project


def _write_test_mesh(target: Path, label: str) -> dict:
    target.write_text(f"o {label}\nv 0 0 0\n", encoding="utf-8")
    return {"vertices": 1, "faces": 0}


class MVPTests(unittest.TestCase):
    def _run_cli(self, args: list[str]) -> tuple[int, list[dict]]:
        stdout = StringIO()
        with patch("sys.argv", ["image2stl.cli", *args]), redirect_stdout(stdout):
            code = cli.main()
        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        return code, [json.loads(line) for line in lines]

    def test_project_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project, project_dir = create_project(base, "MyHeadScan")
            project.scaleMm = 180.0
            project.save(project_dir)
            loaded = load_project(project_dir)
            self.assertEqual(loaded.name, "MyHeadScan")
            self.assertEqual(loaded.scaleMm, 180.0)

    def test_image_validation_insufficient_images(self):
        result = process_command(
            {
                "command": "reconstruct",
                "mode": "local",
                "images": ["a.jpg", "b.jpg"],
                "outputPath": "/tmp/out.obj",
            }
        )
        self.assertEqual(result[0]["errorCode"], "INSUFFICIENT_IMAGES")

    def test_scale_calculation(self):
        factor = calculate_scale_factor((100.0, 150.0, 80.0), 300.0, "longest")
        self.assertEqual(factor, 2.0)

    def test_image_validation_unsupported_format(self):
        result = process_command(
            {
                "command": "reconstruct",
                "mode": "local",
                "images": ["a.jpg", "b.png", "c.bmp"],
                "outputPath": "/tmp/out.obj",
            }
        )
        self.assertEqual(result[0]["errorCode"], "UNSUPPORTED_FILE_FORMAT")

    def test_heif_extension_accepted_in_validation(self):
        """HEIF extension (.heif) should pass image validation like .heic."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "raw.obj"
            with patch("image2stl.engine._run_triposr_local") as run_local:
                run_local.side_effect = lambda _images, target: _write_test_mesh(target, "heif")
                messages = process_command(
                    {
                        "command": "reconstruct",
                        "mode": "local",
                        "images": ["a.jpg", "b.heif", "c.heic"],
                        "outputPath": str(output_path),
                    }
                )
            self.assertEqual(messages[-1]["type"], "success")

    def test_heif_support_registered_by_engine(self):
        """Calling _ensure_heif_support should register HEIC/HEIF with Pillow."""
        from image2stl.engine import _ensure_heif_support
        _ensure_heif_support()
        from PIL import Image
        extensions = Image.registered_extensions()
        self.assertIn(".heic", extensions)
        self.assertIn(".heif", extensions)

    def test_add_images_accepts_heif_extension(self):
        """Project.add_images should accept .heif files."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _, project_dir = create_project(base, "HeifTest")
            image = base / "photo.heif"
            image.write_text("mock_heif_data", encoding="utf-8")
            project = load_project(project_dir)
            copied = project.add_images(project_dir, [image])
            self.assertEqual(len(copied), 1)
            self.assertTrue(copied[0].endswith(".heif"))

    def test_ipc_message_parsing(self):
        payload = parse_json_line('{"command":"repair","inputMesh":"a.obj","outputMesh":"b.obj"}')
        self.assertEqual(payload["command"], "repair")

    def test_reconstruct_progress_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "raw.obj"
            with patch("image2stl.engine._run_triposr_local") as run_local:
                run_local.side_effect = lambda _images, target: _write_test_mesh(target, "local")
                messages = process_command(
                    {
                        "command": "reconstruct",
                        "mode": "local",
                        "images": ["a.jpg", "b.png", "c.heic"],
                        "outputPath": str(output_path),
                    }
                )
            statuses = [m["status"] for m in messages if m.get("type") == "progress"]
            self.assertEqual(
                statuses,
                [
                    "Loading images...",
                    "Running AI reconstruction...",
                    "Repairing mesh...",
                    "Generating preview...",
                ],
            )
            self.assertEqual(messages[-1]["type"], "success")
            self.assertTrue(output_path.exists())

    def test_cloud_mode_requires_api_key(self):
        result = process_command(
            {
                "command": "reconstruct",
                "mode": "cloud",
                "images": ["a.jpg", "b.jpg", "c.jpg"],
                "outputPath": "/tmp/out.obj",
            }
        )
        self.assertEqual(result[0]["errorCode"], "API_ERROR")

    def test_cloud_mode_uses_environment_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "raw.obj"
            with patch.dict(os.environ, {"MESHY_API_KEY": "test-key"}, clear=True):
                with patch("image2stl.engine._run_meshy_cloud") as run_cloud:
                    run_cloud.side_effect = lambda _images, target, _api_key: _write_test_mesh(target, "cloud")
                    messages = process_command(
                        {
                            "command": "reconstruct",
                            "mode": "cloud",
                            "images": ["a.jpg", "b.png", "c.heic"],
                            "outputPath": str(output_path),
                        }
                    )
            self.assertEqual(messages[-1]["type"], "success")
            self.assertTrue(output_path.exists())

    def test_parse_json_line_rejects_non_object(self):
        with self.assertRaises(ValueError):
            parse_json_line(json.dumps([1, 2, 3]))

    def test_scale_command_uses_input_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.stl"
            dst = Path(tmp) / "out.stl"
            src.write_text("solid demo", encoding="utf-8")
            result = process_command(
                {
                    "command": "scale",
                    "inputMesh": str(src),
                    "outputMesh": str(dst),
                    "targetSizeMm": 50,
                    "axis": "width",
                    "inputDimensionsMm": [25.0, 100.0, 25.0],
                }
            )
            self.assertEqual(result[0]["scaleFactor"], 2.0)

    def test_cancel_operation_marks_reconstruct_as_cancelled(self):
        process_command({"command": "cancel", "operationId": "op-1"})
        result = process_command(
            {
                "command": "reconstruct",
                "operationId": "op-1",
                "mode": "local",
                "images": ["a.jpg", "b.png", "c.heic"],
                "outputPath": "/tmp/out.obj",
            }
        )
        self.assertEqual(result[0]["errorCode"], "OPERATION_CANCELLED")

    def test_repair_uses_trimesh_pipeline(self):
        class FakeMesh:
            vertices = [0, 1, 2]
            faces = [0]
            is_watertight = True

            def remove_duplicate_faces(self):
                return None

            def remove_degenerate_faces(self):
                return None

            def remove_unreferenced_vertices(self):
                return None

            def fix_normals(self):
                return None

            def fill_holes(self):
                return None

            def export(self, output_path: str):
                Path(output_path).write_text("solid repaired\nendsolid repaired\n", encoding="utf-8")

        fake_trimesh = types.SimpleNamespace(load=lambda _path, force="mesh": FakeMesh(), Scene=type("Scene", (), {}))
        with patch.dict(sys.modules, {"trimesh": fake_trimesh}, clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "in.stl"
                dst = Path(tmp) / "out.stl"
                src.write_text("solid demo\nendsolid demo\n", encoding="utf-8")
                result = process_command(
                    {
                        "command": "repair",
                        "inputMesh": str(src),
                        "outputMesh": str(dst),
                    }
                )
                self.assertTrue(dst.exists())
        self.assertEqual(result[0]["type"], "success")
        self.assertEqual(result[0]["stats"]["vertices"], 3)

    def test_cli_add_images_updates_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _, project_dir = create_project(base, "WorkflowProject")
            images = []
            for index, suffix in enumerate(("jpg", "png", "heic"), start=1):
                image = base / f"IMG_{index}.{suffix}"
                image.write_text("mock_image_data", encoding="utf-8")
                images.append(image)
            code, output = self._run_cli(["add-images", "--project-dir", str(project_dir), *map(str, images)])
            project = load_project(project_dir)
            self.assertEqual(code, 0)
            self.assertEqual(output[0]["totalImages"], 3)
            self.assertEqual(len(project.images), 3)

    def test_cli_reconstruct_project_wires_project_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _, project_dir = create_project(base, "WorkflowProject")
            image_paths = []
            for index in range(3):
                image = base / f"IMG_{index}.jpg"
                image.write_text("mock_image_data", encoding="utf-8")
                image_paths.append(image)
            self._run_cli(["add-images", "--project-dir", str(project_dir), *map(str, image_paths)])
            with patch.dict(os.environ, {"MESHY_API_KEY": "test-key"}, clear=True):
                with patch("image2stl.engine._run_meshy_cloud") as run_cloud:
                    run_cloud.side_effect = lambda _images, target, _api_key: _write_test_mesh(target, "cloud")
                    code, output = self._run_cli(
                        ["reconstruct-project", "--project-dir", str(project_dir), "--mode", "cloud"]
                    )
            project = load_project(project_dir)
            self.assertEqual(code, 0)
            self.assertEqual(project.reconstructionMode, "cloud")
            self.assertEqual(project.modelPath, "models/raw_reconstruction.obj")
            self.assertEqual(output[-1]["type"], "success")
            self.assertTrue((project_dir / project.modelPath).exists())


if __name__ == "__main__":
    unittest.main()
