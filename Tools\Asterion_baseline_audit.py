
#!/usr/bin/env python3
"""Read-only baseline audit for the Asterion Python project."""

from __future__ import annotations

import argparse
import ast
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "site-packages", "node_modules", "dist", "build",
}
SOURCE_SUFFIXES = {".py"}
DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}

SENSITIVE_NAME_PARTS = (
    "password", "secret", "token", "credential", "api_key", "apikey",
)


def safe_git_info(root: Path) -> dict[str, str]:
    """Read Git metadata only; never modify the repository."""
    result: dict[str, str] = {}
    for key, command in {
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
        "commit": ["git", "rev-parse", "--short", "HEAD"],
    }.items():
        try:
            proc = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            result[key] = (
                proc.stdout.strip()
                if proc.returncode == 0
                else "unavailable"
            )
        except (OSError, subprocess.SubprocessError):
            result[key] = "unavailable"
    return result


def project_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix in SOURCE_SUFFIXES or path.suffix in DB_SUFFIXES:
            files.append(path)
    return sorted(files)


def parse_python(path: Path, root: Path) -> dict[str, Any]:
    relative = str(path.relative_to(root))
    result: dict[str, Any] = {
        "path": relative,
        "parse_ok": False,
        "parse_error": None,
        "imports": [],
        "classes": [],
        "functions": [],
        "entry_point": False,
        "sensitive_identifiers": [],
    }

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
    except (OSError, UnicodeError, SyntaxError) as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["parse_ok"] = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result["imports"].extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            result["imports"].append(prefix)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["functions"].append({
                "name": node.name,
                "line": node.lineno,
            })
        elif isinstance(node, ast.ClassDef):
            result["classes"].append({
                "name": node.name,
                "line": node.lineno,
            })
        elif isinstance(node, ast.Name):
            lowered = node.id.lower()
            if any(part in lowered for part in SENSITIVE_NAME_PARTS):
                result["sensitive_identifiers"].append(node.id)

    result["imports"] = sorted(set(result["imports"]))
    result["sensitive_identifiers"] = sorted(
        set(result["sensitive_identifiers"])
    )
    result["entry_point"] = (
        'if __name__ == "__main__"' in source
        or any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"FastAPI", "Flask"}
            for node in ast.walk(tree)
        )
    )
    return result


def inspect_database(path: Path, root: Path) -> dict[str, Any]:
    """Inspect schema using SQLite read-only mode."""
    item: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "read_only": True,
        "tables": [],
        "error": None,
    }

    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2) as connection:
            connection.execute("PRAGMA query_only = ON")
            tables = connection.execute(
                """
                SELECT name, type
                FROM sqlite_master
                WHERE type IN ('table', 'view')
                ORDER BY type, name
                """
            ).fetchall()

            for name, object_type in tables:
                # Table names come from SQLite's own schema.
                escaped = name.replace('"', '""')
                columns = connection.execute(
                    f'PRAGMA table_info("{escaped}")'
                ).fetchall()
                item["tables"].append({
                    "name": name,
                    "type": object_type,
                    "columns": [
                        {
                            "name": column[1],
                            "type": column[2],
                            "not_null": bool(column[3]),
                            "primary_key": bool(column[5]),
                        }
                        for column in columns
                    ],
                })
    except (sqlite3.Error, OSError, ValueError) as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"

    return item


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Asterion architecture and persistence audit"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--output",
        default="asterion_baseline_report.json",
        help="Report destination (default: a JSON file in current directory)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Project root is not a directory: {root}")

    files = project_files(root)
    python_files = [p for p in files if p.suffix == ".py"]
    database_files = [p for p in files if p.suffix in DB_SUFFIXES]

    python_results = [parse_python(path, root) for path in python_files]
    known_modules = {
        path.stem for path in python_files
    }

    report = {
        "project_root": str(root),
        "git": safe_git_info(root),
        "python_file_count": len(python_files),
        "database_file_count": len(database_files),
        "module_names": sorted(known_modules),
        "python_modules": python_results,
        "sqlite_databases": [
            inspect_database(path, root) for path in database_files
        ],
        "notes": [
            "This report does not execute application code.",
            "Import names are listed, not guaranteed to resolve.",
            "AST inspection cannot prove runtime call paths.",
            "Database inspection uses read-only SQLite connections.",
            "Sensitive identifier matches are heuristic and may be false positives.",
        ],
    }

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Project: {root}")
    print(f"Python files: {len(python_files)}")
    print(f"SQLite files: {len(database_files)}")
    print(f"Report: {output.resolve()}")
    print("\nModules with syntax errors:")
    failures = [m for m in python_results if not m["parse_ok"]]
    if failures:
        for module in failures:
            print(f"  {module['path']}: {module['parse_error']}")
    else:
        print("  None detected.")

    print("\nReview sensitive identifier matches in the JSON report.")
    print("No application modules were imported or executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
