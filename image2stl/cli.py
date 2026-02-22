from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .engine import parse_json_line, process_command
from .project import create_project, load_project


def main() -> int:
    parser = argparse.ArgumentParser(description="Image2STL MVP CLI")
    sub = parser.add_subparsers(dest="action", required=True)

    new_project = sub.add_parser("new")
    new_project.add_argument("--base-dir", type=Path, required=True)
    new_project.add_argument("--name", required=True)

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--json", required=True)

    load_cmd = sub.add_parser("load")
    load_cmd.add_argument("--project-dir", type=Path, required=True)

    add_images_cmd = sub.add_parser("add-images")
    add_images_cmd.add_argument("--project-dir", type=Path, required=True)
    add_images_cmd.add_argument("images", nargs="+", type=Path)

    reconstruct_project_cmd = sub.add_parser("reconstruct-project")
    reconstruct_project_cmd.add_argument("--project-dir", type=Path, required=True)
    reconstruct_project_cmd.add_argument("--mode", choices=("local", "cloud"))
    reconstruct_project_cmd.add_argument("--api-key")

    args = parser.parse_args()

    if args.action == "new":
        project, project_dir = create_project(args.base_dir, args.name)
        print(json.dumps({"projectId": project.projectId, "projectDir": str(project_dir)}))
        return 0

    if args.action == "load":
        print(json.dumps(asdict(load_project(args.project_dir))))
        return 0

    if args.action == "add-images":
        project = load_project(args.project_dir)
        copied = project.add_images(args.project_dir, args.images)
        project.save(args.project_dir)
        print(json.dumps({"projectId": project.projectId, "addedImages": copied, "totalImages": len(project.images)}))
        return 0

    if args.action == "reconstruct-project":
        project = load_project(args.project_dir)
        if args.mode:
            project.reconstructionMode = args.mode
        output_path = (args.project_dir / "models" / "raw_reconstruction.obj").resolve()
        command = {
            "command": "reconstruct",
            "mode": project.reconstructionMode,
            "images": [str((args.project_dir / image).resolve()) for image in project.images],
            "outputPath": str(output_path),
            "projectId": project.projectId,
        }
        if args.api_key:
            command["apiKey"] = args.api_key
        messages = process_command(command)
        for message in messages:
            print(json.dumps(message))
        if messages and messages[-1].get("type") == "success":
            project.modelPath = "models/raw_reconstruction.obj"
            project.save(args.project_dir)
            return 0
        return 1

    command = parse_json_line(args.json)
    for message in process_command(command):
        print(json.dumps(message))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
