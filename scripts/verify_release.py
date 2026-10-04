#!/usr/bin/env python3
"""Validate that this remains a portable, code-and-documentation-only release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_SIZE = 25 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".npy", ".npz", ".h5", ".hdf5",
    ".csv", ".json", ".pkl", ".pickle", ".pdf", ".png", ".jpg",
    ".jpeg", ".svg", ".log",
}
FORBIDDEN_DIRS = {
    "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "checkpoints",
    "logs", "results", "figures", "metrics", "generated", "outputs",
}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".yml", ".yaml", ".cff", ".tsv", ".txt"}
SENSITIVE_PATTERNS = {
    "personal absolute path": re.compile(
        r"/home/" + r"sina|Sina_" + r"GPU|Mem_" + r"Gen_Project"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
}
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files():
    if (ROOT / ".git").is_dir():
        completed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True,
            stdout=subprocess.PIPE,
        )
        for item in completed.stdout.decode("utf-8").split("\0"):
            if item:
                path = ROOT / item
                if path.is_file():
                    yield path
        return
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        yield path


def validate_tree() -> list[str]:
    errors: list[str] = []
    for path in release_files():
        relative = path.relative_to(ROOT)
        lowered_parts = {part.lower() for part in relative.parts[:-1]}
        if lowered_parts & FORBIDDEN_DIRS:
            errors.append(f"forbidden artifact directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact type: {relative}")
        if path.stat().st_size > MAX_SIZE:
            errors.append(f"file exceeds 25 MiB: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} found in {relative}")
    return errors


def validate_markdown_links() -> list[str]:
    errors: list[str] = []
    for path in release_files():
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            linked_path = (path.parent / target).resolve()
            if not linked_path.exists():
                relative = path.relative_to(ROOT)
                errors.append(f"broken local Markdown link in {relative}: {raw_target}")
    return errors


def validate_manifest(source_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    manifest = ROOT / "SOURCE_MANIFEST.tsv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            repository_path = row.get("repository_path", "")
            release_expected = row.get("release_sha256", "")
            original = row.get("original_path", "")
            original_expected = row.get("original_sha256", "")

            release_path = ROOT / repository_path
            if not release_path.is_file():
                errors.append(f"release source missing: {repository_path}")
            elif release_expected and sha256(release_path) != release_expected:
                errors.append(f"release source changed: {repository_path}")

            if source_root and original and original_expected:
                original_path = source_root / original
                if not original_path.is_file():
                    errors.append(f"original source missing: {original}")
                elif sha256(original_path) != original_expected:
                    errors.append(f"original source changed: {original}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        help="Optional original project root for checksum verification",
    )
    args = parser.parse_args()

    errors = validate_tree()
    errors.extend(validate_markdown_links())
    source_root = args.source_root.resolve() if args.source_root else None
    errors.extend(validate_manifest(source_root))

    if errors:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(errors))
    print("Release validation passed.")


if __name__ == "__main__":
    main()
