"""Tests for core/project_manager.py and core/reconstruction_engine.py."""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.project import Project
from core.project_manager import ProjectManager
from core.reconstruction_engine import ReconstructionEngine


class TestProjectManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.base = Path(self.tmp)
        self.pm = ProjectManager(self.base)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_project(self):
        project = self.pm.create_project("TestScan")
        self.assertEqual(project.name, "TestScan")
        self.assertIs(self.pm.current_project, project)
        self.assertTrue(project.project_path.is_dir())

    def test_create_project_with_custom_dir(self):
        custom_dir = self.base / "custom"
        custom_dir.mkdir()
        project = self.pm.create_project("CustomScan", project_dir=custom_dir)
        self.assertEqual(project.project_path.parent, custom_dir)

    def test_open_project(self):
        created = self.pm.create_project("OpenTest")
        pm2 = ProjectManager(self.base)
        loaded = pm2.open_project(created.project_path)
        self.assertEqual(loaded.name, "OpenTest")
        self.assertIs(pm2.current_project, loaded)

    def test_open_nonexistent_project_raises(self):
        with self.assertRaises(Exception):
            self.pm.open_project(self.base / "nonexistent")

    def test_save_current_project(self):
        project = self.pm.create_project("SaveTest")
        project.settings.scale_mm = 250.0
        self.pm.save_current_project()

        loaded = Project.load(project.project_path)
        self.assertAlmostEqual(loaded.settings.scale_mm, 250.0)

    def test_save_no_current_project_is_safe(self):
        self.pm.save_current_project()  # should not raise

    def test_close_current_project(self):
        self.pm.create_project("CloseTest")
        self.pm.close_current_project()
        self.assertIsNone(self.pm.current_project)

    def test_list_projects(self):
        self.pm.create_project("Alpha")
        self.pm.create_project("Beta")
        projects = self.pm.list_projects()
        names = {p.name for p in projects}
        self.assertIn("Alpha", names)
        self.assertIn("Beta", names)

    def test_list_projects_empty_dir(self):
        empty_dir = self.base / "empty"
        empty_dir.mkdir()
        projects = self.pm.list_projects(empty_dir)
        self.assertEqual(projects, [])

    def test_list_projects_sorted_newest_first(self):
        self.pm.create_project("First")
        second = self.pm.create_project("Second")
        # Bump the mtime so sorting is unambiguous regardless of filesystem granularity
        time.sleep(0.01)
        second.project_path.touch()
        projects = self.pm.list_projects()
        self.assertEqual(projects[0], second.project_path)

    def test_delete_project(self):
        project = self.pm.create_project("DeleteTest")
        path = project.project_path
        self.pm.delete_project(path)
        self.assertFalse(path.exists())

    def test_delete_current_project_clears_reference(self):
        project = self.pm.create_project("DeleteCurrent")
        self.pm.delete_project(project.project_path)
        self.assertIsNone(self.pm.current_project)

    def test_delete_nonexistent_is_safe(self):
        self.pm.delete_project(self.base / "ghost")  # should not raise

    def test_default_dir_created_if_missing(self):
        new_dir = self.base / "new_projects"
        self.assertFalse(new_dir.exists())
        ProjectManager(new_dir)
        self.assertTrue(new_dir.is_dir())


class TestReconstructionEngine(unittest.TestCase):
    def test_initial_state(self):
        engine = ReconstructionEngine()
        self.assertFalse(engine.is_running)

    def test_reconstruct_calls_on_success(self):
        engine = ReconstructionEngine()
        results = {}
        done = threading.Event()

        def _fake_process(command):
            return [{"type": "success", "outputPath": "/tmp/out.obj", "stats": {}}]

        def _on_success(path, stats):
            results["path"] = path
            done.set()

        with patch("core.reconstruction_engine.process_command", side_effect=_fake_process):
            engine.reconstruct(
                images=["a.jpg", "b.jpg", "c.jpg"],
                output_path=Path("/tmp/out.obj"),
                on_success=_on_success,
            )
            done.wait(timeout=5)

        self.assertEqual(results.get("path"), "/tmp/out.obj")

    def test_reconstruct_calls_on_error(self):
        engine = ReconstructionEngine()
        results = {}
        done = threading.Event()

        def _fake_process(command):
            return [{"type": "error", "errorCode": "RECONSTRUCTION_FAILED", "message": "fail"}]

        def _on_error(code, message):
            results["code"] = code
            done.set()

        with patch("core.reconstruction_engine.process_command", side_effect=_fake_process):
            engine.reconstruct(
                images=["a.jpg", "b.jpg", "c.jpg"],
                output_path=Path("/tmp/out.obj"),
                on_error=_on_error,
            )
            done.wait(timeout=5)

        self.assertEqual(results.get("code"), "RECONSTRUCTION_FAILED")

    def test_reconstruct_calls_on_progress(self):
        engine = ReconstructionEngine()
        progress_values = []
        done = threading.Event()

        def _fake_process(command):
            return [
                {"type": "progress", "progress": 0.5, "status": "Running…"},
                {"type": "success", "outputPath": "/tmp/out.obj", "stats": {}},
            ]

        def _on_progress(fraction, status, estimated):
            progress_values.append(fraction)

        def _on_success(path, stats):
            done.set()

        with patch("core.reconstruction_engine.process_command", side_effect=_fake_process):
            engine.reconstruct(
                images=["a.jpg", "b.jpg", "c.jpg"],
                output_path=Path("/tmp/out.obj"),
                on_progress=_on_progress,
                on_success=_on_success,
            )
            done.wait(timeout=5)

        self.assertIn(0.5, progress_values)

    def test_thread_cleared_after_completion(self):
        engine = ReconstructionEngine()
        done = threading.Event()

        def _fake_process(command):
            return [{"type": "success", "outputPath": "/tmp/out.obj", "stats": {}}]

        def _on_success(path, stats):
            done.set()

        with patch("core.reconstruction_engine.process_command", side_effect=_fake_process):
            engine.reconstruct(
                images=["a.jpg", "b.jpg", "c.jpg"],
                output_path=Path("/tmp/out.obj"),
                on_success=_on_success,
            )
            done.wait(timeout=5)

        # Poll until the thread finishes its finally block
        deadline = time.monotonic() + 5.0
        while engine._thread is not None and time.monotonic() < deadline:
            time.sleep(0.005)

        self.assertFalse(engine.is_running)
        self.assertIsNone(engine._thread)
        self.assertIsNone(engine._current_operation_id)

    def test_second_reconstruct_blocked_while_running(self):
        engine = ReconstructionEngine()
        started = threading.Event()
        finish = threading.Event()

        def _slow_process(command):
            started.set()
            finish.wait()
            return [{"type": "success", "outputPath": "/tmp/out.obj", "stats": {}}]

        try:
            with patch("core.reconstruction_engine.process_command", side_effect=_slow_process):
                engine.reconstruct(
                    images=["a.jpg", "b.jpg", "c.jpg"],
                    output_path=Path("/tmp/out.obj"),
                )
                started.wait(timeout=5)
                self.assertTrue(engine.is_running)

                # Second call should be silently ignored
                first_thread = engine._thread
                engine.reconstruct(
                    images=["a.jpg", "b.jpg", "c.jpg"],
                    output_path=Path("/tmp/out2.obj"),
                )
                self.assertIs(engine._thread, first_thread)
        finally:
            finish.set()

    def test_callback_exception_does_not_crash_thread(self):
        """A failing on_progress callback must not prevent on_success from running."""
        engine = ReconstructionEngine()
        done = threading.Event()

        def _fake_process(command):
            return [
                {"type": "progress", "progress": 0.5, "status": "x"},
                {"type": "success", "outputPath": "/tmp/out.obj", "stats": {}},
            ]

        def _bad_progress(f, s, e):
            raise RuntimeError("callback error")

        def _on_success(path, stats):
            done.set()

        with patch("core.reconstruction_engine.process_command", side_effect=_fake_process):
            engine.reconstruct(
                images=["a.jpg", "b.jpg", "c.jpg"],
                output_path=Path("/tmp/out.obj"),
                on_progress=_bad_progress,
                on_success=_on_success,
            )
            done.wait(timeout=5)

        self.assertTrue(done.is_set(), "on_success should still fire after bad on_progress")

    def test_cancel_sends_cancel_command(self):
        engine = ReconstructionEngine()
        commands_seen = []

        def _fake_process(command):
            commands_seen.append(command.get("command"))
            return [{"type": "success", "outputPath": "/tmp/out.obj", "stats": {}}]

        with patch("core.reconstruction_engine.process_command", side_effect=_fake_process):
            engine._current_operation_id = "test-op-123"
            engine.cancel()

        self.assertIn("cancel", commands_seen)


if __name__ == "__main__":
    unittest.main()
