from __future__ import annotations

import argparse
import json
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

    args = parser.parse_args()

    if args.action == "new":
        project, project_dir = create_project(args.base_dir, args.name)
        print(json.dumps({"projectId": project.projectId, "projectDir": str(project_dir)}))
        return 0

    if args.action == "load":
        print(json.dumps(load_project(args.project_dir).__dict__))
        return 0

    command = parse_json_line(args.json)
    for message in process_command(command):
        print(json.dumps(message))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
