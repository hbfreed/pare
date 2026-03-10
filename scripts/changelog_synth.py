#!/usr/bin/env python3
"""Generate a Keep-a-Changelog-style CHANGELOG.md from git history.

Usage:
    python scripts/changelog_synth.py [--repo PATH] [--output PATH]

No external dependencies — uses only the Python standard library.
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

# Maps leading verbs in commit messages to changelog categories.
# Order matters: first match wins.
CATEGORY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^add(ed|s|ing)?\b", re.I), "Added"),
    (re.compile(r"^(implement|introduce|create|wrote)\b", re.I), "Added"),
    (re.compile(r"^fix(ed|es|ing)?\b", re.I), "Fixed"),
    (re.compile(r"^remov(e[ds]?|ing)\b", re.I), "Removed"),
    (re.compile(r"^delet(e[ds]?|ing)\b", re.I), "Removed"),
    (re.compile(r"^deprecat(e[ds]?|ing)\b", re.I), "Deprecated"),
]

# Preferred display order for categories.
CATEGORY_ORDER = ["Added", "Changed", "Deprecated", "Fixed", "Removed"]

SEPARATOR = "---SEP---"
LOG_FORMAT = f"%H{SEPARATOR}%aI{SEPARATOR}%s"


def get_commits(repo_path: str) -> list[dict]:
    """Run git log and return a list of parsed commit dicts."""
    result = subprocess.run(
        ["git", "log", f"--format={LOG_FORMAT}", "--reverse"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commits = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(SEPARATOR, maxsplit=2)
        if len(parts) != 3:
            continue
        hash_, date_str, message = parts
        date = datetime.fromisoformat(date_str)
        commits.append({"hash": hash_, "date": date, "message": message.strip()})
    return commits


def categorize(message: str) -> str:
    """Determine the changelog category from a commit message."""
    for pattern, category in CATEGORY_RULES:
        if pattern.search(message):
            return category
    return "Changed"


def group_by_month(commits: list[dict]) -> dict[str, list[dict]]:
    """Group commits by YYYY-MM key, most recent month first."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for commit in commits:
        key = commit["date"].strftime("%Y-%m")
        groups[key].append(commit)
    # Return sorted descending by month key.
    return dict(sorted(groups.items(), reverse=True))


def format_changelog(grouped: dict[str, list[dict]]) -> str:
    """Render grouped commits as Keep-a-Changelog markdown."""
    lines = [
        "# Changelog",
        "",
        "All notable changes to this project will be documented in this file.",
        "",
        "The format is based on [Keep a Changelog](https://keepachangelog.com/).",
        "",
    ]

    for month_key, commits in grouped.items():
        # Section heading: "## 2026-03 (March 2026)"
        dt = datetime.strptime(month_key, "%Y-%m")
        heading = f"## {month_key} ({dt.strftime('%B %Y')})"
        lines.append(heading)
        lines.append("")

        # Sub-group by category.
        by_category: dict[str, list[dict]] = defaultdict(list)
        for commit in commits:
            cat = categorize(commit["message"])
            by_category[cat].append(commit)

        for cat in CATEGORY_ORDER:
            if cat not in by_category:
                continue
            lines.append(f"### {cat}")
            lines.append("")
            for c in by_category[cat]:
                short_hash = c["hash"][:7]
                lines.append(f"- {c['message']} (`{short_hash}`)")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CHANGELOG.md from git log")
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to git repository (default: current directory)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: CHANGELOG.md in repo root)",
    )
    args = parser.parse_args()

    output_path = args.output or f"{args.repo}/CHANGELOG.md"

    commits = get_commits(args.repo)
    if not commits:
        print("No commits found.", file=sys.stderr)
        sys.exit(1)

    grouped = group_by_month(commits)
    changelog = format_changelog(grouped)

    with open(output_path, "w") as f:
        f.write(changelog)

    print(f"Generated {output_path} ({len(commits)} commits)")


if __name__ == "__main__":
    main()
