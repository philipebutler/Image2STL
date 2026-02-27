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
from image2stl.engine import calculate_scale_factor, check_image_quality, parse_json_line, process_command
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
        """Calling _ensure_heif_support should register HEIC/HEIF with Pillow when pillow-heif is available."""
        try:
            import pillow_heif  # noqa: F401
        except ImportError:
            self.skipTest("pillow-heif not installed")
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
                    "Checking Python dependencies...",
                    "Checking TripoSR model cache...",
                    "Loading/downloading TripoSR model weights...",
                    "Repairing mesh...",
                    "Generating preview...",
                ],
            )
            self.assertEqual(messages[-1]["type"], "success")
            self.assertTrue(output_path.exists())

    def test_check_environment_reports_local_dependencies(self):
        result = process_command({"command": "check_environment", "mode": "local"})
        self.assertEqual(result[0]["type"], "success")
        self.assertEqual(result[0]["command"], "check_environment")
        self.assertIn("python", result[0])
        self.assertIn("local", result[0])
        self.assertIn("dependencies", result[0]["local"])
        self.assertIn("model", result[0])

    def test_reconstruct_returns_dependency_error_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "raw.obj"
            with patch("image2stl.engine._ensure_local_dependencies") as ensure_local:
                from image2stl.engine import MissingDependenciesError

                ensure_local.side_effect = MissingDependenciesError([
                    {"module": "torch", "package": "torch"},
                ])
                messages = process_command(
                    {
                        "command": "reconstruct",
                        "mode": "local",
                        "images": ["a.jpg", "b.png", "c.heic"],
                        "outputPath": str(output_path),
                    }
                )

        self.assertEqual(messages[0]["type"], "error")
        self.assertEqual(messages[0]["errorCode"], "PYTHON_DEPENDENCIES_MISSING")
        self.assertEqual(messages[0]["missingDependencies"][0]["module"], "torch")

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
            src.write_text(
                """solid demo
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid demo
""",
                encoding="utf-8",
            )
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
            self.assertTrue(dst.exists())

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

    def test_webp_extension_accepted_in_validation(self):
        """WebP extension (.webp) should pass image validation."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "raw.obj"
            with patch("image2stl.engine._run_triposr_local") as run_local:
                run_local.side_effect = lambda _images, target: _write_test_mesh(target, "webp")
                messages = process_command(
                    {
                        "command": "reconstruct",
                        "mode": "local",
                        "images": ["a.jpg", "b.webp", "c.png"],
                        "outputPath": str(output_path),
                    }
                )
            self.assertEqual(messages[-1]["type"], "success")

    def test_avif_extension_accepted_in_validation(self):
        """AVIF extension (.avif) should pass image validation."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "raw.obj"
            with patch("image2stl.engine._run_triposr_local") as run_local:
                run_local.side_effect = lambda _images, target: _write_test_mesh(target, "avif")
                messages = process_command(
                    {
                        "command": "reconstruct",
                        "mode": "local",
                        "images": ["a.avif", "b.jpg", "c.png"],
                        "outputPath": str(output_path),
                    }
                )
            self.assertEqual(messages[-1]["type"], "success")

    def test_add_images_accepts_webp_extension(self):
        """Project.add_images should accept .webp files."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _, project_dir = create_project(base, "WebpTest")
            image = base / "photo.webp"
            image.write_text("mock_webp_data", encoding="utf-8")
            project = load_project(project_dir)
            copied = project.add_images(project_dir, [image])
            self.assertEqual(len(copied), 1)
            self.assertTrue(copied[0].endswith(".webp"))

    def test_check_image_quality_low_resolution(self):
        """check_image_quality should warn about images smaller than 512x512."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "small.png"
            Image.new("RGB", (256, 256), (128, 128, 128)).save(str(small))
            warnings = check_image_quality([str(small)])
            issues = [w["issue"] for w in warnings]
            self.assertIn("low_resolution", issues)

    def test_check_image_quality_ok(self):
        """check_image_quality should return no warnings for a large sharp image."""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("Pillow or numpy not installed")
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.png"
            # Create an image with high-frequency content (not blurry)
            arr = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
            Image.fromarray(arr).save(str(good))
            warnings = check_image_quality([str(good)])
            self.assertEqual(len(warnings), 0)

    def test_check_images_command(self):
        """The check_images command should return quality warnings."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "tiny.png"
            Image.new("RGB", (100, 100), (50, 50, 50)).save(str(small))
            result = process_command({"command": "check_images", "images": [str(small)]})
            self.assertEqual(result[0]["type"], "success")
            self.assertTrue(len(result[0]["warnings"]) > 0)

    def test_repair_includes_watertight_validation(self):
        """Repair result should include a validation dict with pass/fail."""
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

            def split(self):
                return [self]

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
        self.assertEqual(result[0]["type"], "success")
        self.assertIn("validation", result[0])
        self.assertEqual(result[0]["validation"]["watertight"], True)
        self.assertEqual(result[0]["validation"]["result"], "pass")


    def test_preprocess_images_command_success(self):
        """preprocess_images engine command should return processed file paths on success."""
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            images_dir = project_dir / "images"
            images_dir.mkdir()
            img = images_dir / "photo1.jpg"
            img.write_bytes(b"fake_image_data")
            processed_dir = project_dir / "preview" / "processed"
            processed_dir.mkdir(parents=True)

            def _fake_preprocess(source, output_dir, **kwargs):
                out = output_dir / f"{source.stem}_processed.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"fake_rgba")
                return out

            import types as _types
            fake_pre = _types.ModuleType("image2stl.preprocess")
            fake_pre.preprocess_image = _fake_preprocess
            with patch.dict(sys.modules, {"image2stl.preprocess": fake_pre}):
                result = process_command({
                    "command": "preprocess_images",
                    "images": [str(img)],
                    "outputDir": str(processed_dir),
                    "strength": 0.5,
                })
            self.assertEqual(result[0]["type"], "success")
            self.assertEqual(result[0]["command"], "preprocess_images")
            self.assertEqual(len(result[0]["processedImages"]), 1)

    def test_preprocess_images_handles_missing_rembg(self):
        """preprocess_images should return REMBG_UNAVAILABLE when rembg is not installed."""
        import sys

        # Simulate rembg being absent by making the preprocess module raise ImportError
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp) / "preview" / "processed"

            def _raise_import(*args, **kwargs):
                raise ImportError("rembg is not installed")

            import types as _types
            fake_pre = _types.ModuleType("image2stl.preprocess")
            fake_pre.preprocess_image = _raise_import
            with patch.dict(sys.modules, {"image2stl.preprocess": fake_pre}):
                result = process_command({
                    "command": "preprocess_images",
                    "images": ["a.jpg"],
                    "outputDir": str(processed_dir),
                })
            self.assertEqual(result[0]["type"], "error")
            self.assertEqual(result[0]["errorCode"], "REMBG_UNAVAILABLE")

    def test_cli_preprocess_images_writes_preview_processed(self):
        """preprocess-images CLI command should invoke engine and report success."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _, project_dir = create_project(base, "PreprocessTest")
            img = base / "photo.jpg"
            img.write_bytes(b"fake_image_data")
            self._run_cli(["add-images", "--project-dir", str(project_dir), str(img)])

            processed_dir = project_dir / "preview" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            expected_out = processed_dir / "photo_dummyhash_processed.png"

            def _fake_preprocess(source, output_dir, **kwargs):
                hash_part = "dummyhash"
                out = output_dir / f"{source.stem}_{hash_part}_processed.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"fake_rgba")
                return out

            import sys
            import types as _types
            fake_pre = _types.ModuleType("image2stl.preprocess")
            fake_pre.preprocess_image = _fake_preprocess
            with patch.dict(sys.modules, {"image2stl.preprocess": fake_pre}):
                code, output = self._run_cli([
                    "preprocess-images",
                    "--project-dir", str(project_dir),
                ])
            self.assertEqual(code, 0)
            self.assertEqual(output[-1]["type"], "success")
            self.assertEqual(output[-1]["command"], "preprocess_images")

    def test_reconstruct_project_uses_processed_images_when_selected(self):
        """reconstruct-project --preprocess-source processed should use processed images."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _, project_dir = create_project(base, "ProcessedSrcTest")
            image_paths = []
            for index in range(3):
                img = base / f"IMG_{index}.jpg"
                img.write_bytes(b"fake_image_data")
                image_paths.append(img)
            self._run_cli(["add-images", "--project-dir", str(project_dir), *map(str, image_paths)])

            # Write fake processed images under preview/processed
            processed_dir = project_dir / "preview" / "processed"
            processed_dir.mkdir(parents=True)
            proc_images = []
            for index in range(3):
                p = processed_dir / f"IMG_{index}_processed.png"
                p.write_bytes(b"fake_rgba")
                proc_images.append(str(p))

            # Patch the project to have processedImages so CLI can pick them up
            proj = load_project(project_dir)
            proj.processedImages = [f"preview/processed/IMG_{i}_processed.png" for i in range(3)]
            proj.save(project_dir)

            captured_images = {}

            def _fake_cloud(images, target, api_key):
                captured_images["images"] = images
                return _write_test_mesh(target, "cloud")

            with patch.dict(os.environ, {"MESHY_API_KEY": "test-key"}, clear=True):
                with patch("image2stl.engine._run_meshy_cloud", side_effect=_fake_cloud):
                    code, output = self._run_cli([
                        "reconstruct-project",
                        "--project-dir", str(project_dir),
                        "--mode", "cloud",
                        "--preprocess-source", "processed",
                    ])
            self.assertEqual(code, 0)
            # The reconstruction should have used the processed images list
            self.assertTrue(
                all("processed" in p or p.endswith(".png") for p in captured_images.get("images", [])),
                f"Expected processed image paths, got: {captured_images.get('images')}",
            )

    def test_project_roundtrip_persists_preprocess_settings(self):
        """Project save/load should round-trip all preprocess settings fields."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _, project_dir = create_project(base, "PreprocessSettingsTest")
            project = load_project(project_dir)
            project.settings["auto_isolate_enabled"] = True
            project.settings["preprocess_strength"] = 0.8
            project.settings["preprocess_source_mode"] = "processed"
            project.settings["hole_fill_enabled"] = False
            project.settings["island_removal_threshold"] = 250
            project.settings["crop_padding"] = 20
            project.save(project_dir)
            loaded = load_project(project_dir)
            self.assertEqual(loaded.settings.get("auto_isolate_enabled"), True)
            self.assertAlmostEqual(loaded.settings.get("preprocess_strength"), 0.8)
            self.assertEqual(loaded.settings.get("preprocess_source_mode"), "processed")
            self.assertEqual(loaded.settings.get("hole_fill_enabled"), False)
            self.assertEqual(loaded.settings.get("island_removal_threshold"), 250)
            self.assertEqual(loaded.settings.get("crop_padding"), 20)


if __name__ == "__main__":
    unittest.main()
