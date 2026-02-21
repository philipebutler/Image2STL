import json
import tempfile
import unittest
from pathlib import Path

from image2stl.engine import calculate_scale_factor, parse_json_line, process_command
from image2stl.project import create_project, load_project


class MVPTests(unittest.TestCase):
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

    def test_ipc_message_parsing(self):
        payload = parse_json_line('{"command":"repair","inputMesh":"a.obj","outputMesh":"b.obj"}')
        self.assertEqual(payload["command"], "repair")

    def test_reconstruct_progress_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "raw.obj"
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


if __name__ == "__main__":
    unittest.main()
