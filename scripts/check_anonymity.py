#!/usr/bin/env python3
"""Fail when a release tree contains common identity or secret artifacts."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh"
}
FORBIDDEN = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
    "local macOS path": re.compile(r"/Users/[^/]+/"),
    "private home path": re.compile(r"/(?:home|root)/[^\s'\"]+"),
    "private shared-storage path": re.compile(r"/pfs/[^\s'\"]+"),
    "SSH command or host": re.compile(r"(?:ssh\s+-|scp\s+.*@|@[A-Za-z0-9.-]+:\d+)"),
}
FORBIDDEN_NAMES = {".git", ".DS_Store", ".env", "wandb", "outputs", "output", "logs"}


def main():
    findings = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative == Path("scripts/check_anonymity.py"):
            continue
        # Git metadata belongs to the local checkout and is never part of a
        # GitHub working tree or source archive produced with `git archive`.
        if ".git" in relative.parts:
            continue
        if any(part in FORBIDDEN_NAMES or part == "__pycache__" for part in relative.parts):
            findings.append(f"forbidden artifact: {relative}")
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > 100 * 1024 * 1024:
            findings.append(f"file larger than 100 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")
    if findings:
        print("Anonymity check failed:")
        print("\n".join(f"- {item}" for item in findings))
        return 1
    print("Anonymity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
