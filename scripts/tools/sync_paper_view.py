#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path, PurePosixPath


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]

DEFAULT_ROOT = REPO.parents[1]
ROOT = Path(
    os.environ.get("MTPATCHER_ROOT", str(DEFAULT_ROOT))
).resolve()

MANIFEST = REPO / "manifests/paper_view.json"
VIEW = ROOT / "paper_view"
NEXT = ROOT / ".paper_view.next"

ALLOWED_TOP = {
    "experiments",
    "analysis",
    "results",
    "tensorboard",
}

# These concepts must never become paper-facing names.
FORBIDDEN_VIEW_TOKENS = {
    "smoke",
    "invalid",
    "failed",
    "failure",
    "preflight",
    "debug",
    "closure",
    "reconciliation",
    "demo",
    "human6565",
    "prereg",
    "gate",
}

# Physical historical directory names are allowed to be ugly.
# Only unmistakably invalid engineering artifacts are rejected.
FORBIDDEN_TARGET_SUBSTRINGS = (
    "smoke",
    "invalid",
    "failed_launch",
    "duplicate_aborted",
)

ALLOWED_REGULAR_FILES = {
    "README.md",
    ".paper_view_generated",
}


def die(msg):
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def load_manifest():
    if not MANIFEST.is_file():
        die(f"manifest missing: {MANIFEST}")

    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as e:
        die(f"cannot parse manifest: {e}")

    if data.get("schema_version") != 1:
        die("unsupported schema_version")

    entries = data.get("entries")

    if not isinstance(entries, list):
        die("entries must be a list")

    return data, entries


def validate_relative(value, field):
    p = PurePosixPath(value)

    if p.is_absolute():
        die(f"{field} must be relative: {value}")

    if ".." in p.parts:
        die(f"{field} may not contain '..': {value}")

    if not p.parts:
        die(f"empty {field}")

    return p


def normalized_tokens(s):
    return {
        x
        for x in re.split(r"[^a-z0-9]+", s.lower())
        if x
    }


def manifest_sha256():
    h = hashlib.sha256()
    h.update(MANIFEST.read_bytes())
    return h.hexdigest()


def validate_manifest(entries):
    seen = set()

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            die(f"entry {i} is not an object")

        view_s = entry.get("view")
        target_s = entry.get("target")

        if not isinstance(view_s, str):
            die(f"entry {i}: invalid view")

        if not isinstance(target_s, str):
            die(f"entry {i}: invalid target")

        view = validate_relative(view_s, "view")
        target = validate_relative(target_s, "target")

        if view.parts[0] not in ALLOWED_TOP:
            die(
                f"entry {i}: illegal paper-view root "
                f"{view.parts[0]!r}"
            )

        bad = normalized_tokens(view_s) & FORBIDDEN_VIEW_TOKENS

        if bad:
            die(
                f"entry {i}: forbidden paper-view token(s) "
                f"{sorted(bad)} in {view_s}"
            )

        if view_s in seen:
            die(f"duplicate view alias: {view_s}")

        seen.add(view_s)

        target_lower = target_s.lower()

        for bad_target in FORBIDDEN_TARGET_SUBSTRINGS:
            if bad_target in target_lower:
                die(
                    f"entry {i}: target appears non-paper-final "
                    f"({bad_target}): {target_s}"
                )

        physical = ROOT / target

        if not physical.exists():
            die(
                f"entry {i}: target missing: "
                f"{physical}"
            )

    print(f"MANIFEST_VALID=PASS entries={len(entries)}")


def expected_parent_dirs(entries):
    dirs = set()

    for entry in entries:
        p = PurePosixPath(entry["view"])

        for parent in p.parents:
            if str(parent) != ".":
                dirs.add(str(parent))

    return dirs


def check_tree(root, entries, allow_missing_marker=False):
    errors = []

    if not root.is_dir():
        errors.append(f"paper view missing: {root}")
        return errors

    expected_links = {
        entry["view"]: (ROOT / entry["target"]).resolve()
        for entry in entries
    }

    expected_dirs = expected_parent_dirs(entries)

    # Expected links.
    for rel, target in expected_links.items():
        p = root / rel

        if not p.is_symlink():
            errors.append(f"missing/non-symlink entry: {p}")
            continue

        if not p.exists():
            errors.append(f"broken link: {p}")
            continue

        try:
            actual = p.resolve(strict=True)
        except Exception as e:
            errors.append(f"cannot resolve {p}: {e}")
            continue

        if actual != target:
            errors.append(
                f"wrong target: {p}\n"
                f"  actual:   {actual}\n"
                f"  expected: {target}"
            )

    # Nothing unmanaged may silently appear.
    for p in root.rglob("*"):
        rel = str(p.relative_to(root))

        if p.is_symlink():
            if rel not in expected_links:
                errors.append(f"unmanaged symlink: {p}")
            elif not p.exists():
                errors.append(f"broken symlink: {p}")
            continue

        if p.is_dir():
            if rel not in expected_dirs:
                errors.append(f"unmanaged directory: {p}")
            continue

        if p.is_file():
            if p.name not in ALLOWED_REGULAR_FILES:
                errors.append(f"unmanaged regular file: {p}")

    marker = root / ".paper_view_generated"

    if not marker.is_file() and not allow_missing_marker:
        errors.append(f"generated marker missing: {marker}")

    return errors


def write_generated_metadata(root):
    readme = root / "README.md"

    readme.write_text(
        """# MT-PATCHER Paper View

GENERATED DIRECTORY — DO NOT EDIT MANUALLY.

Source of truth:

    repo/MT-Patcher-Reproduction-Ascend/manifests/paper_view.json

Regenerate:

    python scripts/tools/sync_paper_view.py --apply

Verify:

    python scripts/tools/sync_paper_view.py --check

This view exposes only paper-facing final experiment assets.
Development, smoke, failed attempts, debug runs, closure experiments,
preflight runs, demos, and superseded versions stay outside this view.
""",
        encoding="utf-8",
    )

    marker = {
        "schema_version": 1,
        "manifest": str(MANIFEST),
        "manifest_sha256": manifest_sha256(),
    }

    (root / ".paper_view_generated").write_text(
        json.dumps(marker, indent=2) + "\n",
        encoding="utf-8",
    )


def build_tree(root, entries):
    root.mkdir(parents=True, exist_ok=False)

    for entry in entries:
        dst = root / entry["view"]
        src = (ROOT / entry["target"]).resolve()

        dst.parent.mkdir(parents=True, exist_ok=True)

        os.symlink(
            str(src),
            str(dst),
            target_is_directory=src.is_dir(),
        )

    write_generated_metadata(root)


def apply(entries):
    # Refuse to destroy an unexpected manually-maintained directory.
    if VIEW.exists():
        errors = check_tree(
            VIEW,
            entries,
            allow_missing_marker=True,
        )

        if errors:
            print("REFUSE_TO_REPLACE_EXISTING_PAPER_VIEW")
            for e in errors:
                print(f"  {e}")
            raise SystemExit(1)

    if NEXT.exists():
        if NEXT.is_symlink():
            NEXT.unlink()
        elif NEXT.is_dir():
            shutil.rmtree(NEXT)
        else:
            die(f"unexpected temp object: {NEXT}")

    build_tree(NEXT, entries)

    errors = check_tree(
        NEXT,
        entries,
        allow_missing_marker=False,
    )

    if errors:
        print("GENERATED_TREE_CHECK=FAIL")
        for e in errors:
            print(f"  {e}")
        raise SystemExit(1)

    if VIEW.exists():
        shutil.rmtree(VIEW)

    NEXT.rename(VIEW)

    final_errors = check_tree(
        VIEW,
        entries,
        allow_missing_marker=False,
    )

    if final_errors:
        print("FINAL_CHECK=FAIL")
        for e in final_errors:
            print(f"  {e}")
        raise SystemExit(1)

    print(f"PAPER_VIEW_APPLY=PASS root={VIEW}")


def check(entries):
    errors = check_tree(
        VIEW,
        entries,
        allow_missing_marker=False,
    )

    if errors:
        print("PAPER_VIEW_CHECK=FAIL")

        for e in errors:
            print(f"  {e}")

        raise SystemExit(1)

    print(f"PAPER_VIEW_CHECK=PASS entries={len(entries)}")


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--check",
        action="store_true",
        help="validate manifest and existing paper view",
    )

    mode.add_argument(
        "--apply",
        action="store_true",
        help="regenerate paper view from manifest",
    )

    args = parser.parse_args()

    _, entries = load_manifest()

    validate_manifest(entries)

    if args.apply:
        apply(entries)
    else:
        check(entries)


if __name__ == "__main__":
    main()
