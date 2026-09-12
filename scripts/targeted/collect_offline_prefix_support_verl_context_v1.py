#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/workspace/mtpatcher")
REPO = ROOT / "repo/MT-Patcher-Reproduction-Ascend"
VERL = ROOT / "repo/verl-v0.9.0"

KEYWORDS = (
    "forward_kl_topk",
    "distillation",
    "generate_sequences",
    "update_actor",
    'batch["responses"]',
    "batch['responses']",
    "responses",
    "teacher_log",
    "topk",
    "DataProto",
)
MAX_FILE_BYTES = 600_000
MAX_SOURCE_FILES = 24


def cmd(args: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        check=False,
    )
    return f"$ {' '.join(args)}\nrc={p.returncode}\n{p.stdout}\n"


def append_file(dst, path: Path) -> None:
    dst.write("\n\n" + "#" * 78 + "\n")
    dst.write(f"FILE: {path}\n")
    dst.write("#" * 78 + "\n")

    if not path.is_file():
        dst.write("MISSING\n")
        return

    size = path.stat().st_size
    dst.write(f"bytes={size}\n")
    if size > MAX_FILE_BYTES:
        dst.write("SKIPPED_TOO_LARGE\n")
        return

    dst.write(path.read_text(encoding="utf-8", errors="replace"))
    dst.write("\n")


def source_score(path: Path) -> tuple[int, dict[str, int]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0, {}

    lower = text.lower()
    counts = {k: lower.count(k.lower()) for k in KEYWORDS}
    score = (
        20 * counts["forward_kl_topk"]
        + 10 * counts["distillation"]
        + 8 * counts["generate_sequences"]
        + 8 * counts["update_actor"]
        + 5 * (counts['batch["responses"]'] + counts["batch['responses']"])
        + 2 * counts["teacher_log"]
        + counts["topk"]
        + counts["DataProto"]
        + min(counts["responses"], 20)
    )
    return score, counts


def find_agent_files() -> list[Path]:
    candidates = [
        ROOT / "AGENT3.md",
        REPO / "AGENT3.md",
        REPO.parent / "AGENT3.md",
        ROOT / "repo/AGENT3.md",
    ]
    return sorted({p for p in candidates if p.is_file()})


def parquet_schema_text(path: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        return f"pyarrow import failed: {e!r}\n"

    if not path.is_file():
        return f"MISSING {path}\n"

    pf = pq.ParquetFile(path)
    table = pq.read_table(path).slice(0, 3)

    out = [
        f"path={path}",
        f"rows={pf.metadata.num_rows}",
        "schema:",
        str(table.schema),
        "first3_python_repr:",
    ]

    rows = table.to_pylist()
    for i, row in enumerate(rows):
        safe = {}
        for k, v in row.items():
            r = repr(v)
            safe[k] = r if len(r) <= 2500 else r[:2500] + "...<truncated>"
        out.append(f"ROW[{i}]={json.dumps(safe, ensure_ascii=False, indent=2)}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    if VERL.is_dir():
        for path in VERL.rglob("*"):
            if (
                path.is_file()
                and path.suffix in {".py", ".yaml", ".yml"}
                and path.stat().st_size <= MAX_FILE_BYTES
            ):
                score, counts = source_score(path)
                if score > 0:
                    candidates.append((score, str(path), path, counts))

    candidates.sort(key=lambda x: (-x[0], x[1]))

    explicit = [
        VERL / "verl/trainer/main_ppo.py",
        VERL / "verl/trainer/ppo/ray_trainer.py",
        REPO / "scripts/targeted/offline_prefix_support_prepare_v1.py",
        REPO / "recipes/targeted/run_offline_prefix_support_prepare64_v1.sh",
        REPO / "manifests/experiments/targeted/08_offline_prefix_support_replay.json",
        run / "prepare_manifest.json",
        run / "prepare64_summary.json",
        run / "teacher_signal/chemistry_summary_first64.json",
        run / "teacher_signal/idiom_summary_first64.json",
        run / "paired/chemistry_hashes_first64.json",
        run / "paired/idiom_hashes_first64.json",
    ]

    resolved = [
        ROOT
        / "runs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b_20260903_034058"
        / "resolved_config.yaml",
        ROOT
        / "runs/opd/opd_recovery_20260905_v2"
        / "resolved_config.yaml",
    ]

    seen: set[str] = set()

    with out.open("w", encoding="utf-8") as dst:
        dst.write("OFFLINE PREFIX-SUPPORT REPLAY / LOCAL VERL PATCH CONTEXT\n")

        dst.write("\n" + "#" * 78 + "\n")
        dst.write("REPO STATE\n")
        dst.write("#" * 78 + "\n")
        dst.write(cmd(["git", "-C", str(REPO), "status", "--short"]))
        dst.write(cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"]))
        dst.write(cmd(["git", "-C", str(REPO), "diff", "--check"]))

        dst.write("\n" + "#" * 78 + "\n")
        dst.write("VERL STATE\n")
        dst.write("#" * 78 + "\n")
        dst.write(cmd(["git", "-C", str(VERL), "status", "--short"]))
        dst.write(cmd(["git", "-C", str(VERL), "rev-parse", "HEAD"]))
        dst.write(cmd(["git", "-C", str(VERL), "diff", "--stat"]))

        dst.write("\n" + "#" * 78 + "\n")
        dst.write("AGENT3 DISCOVERY\n")
        dst.write("#" * 78 + "\n")
        agents = find_agent_files()
        if not agents:
            dst.write("NO_AGENT3_FOUND_UNDER_ROOT_DEPTH5\n")
        for p in agents:
            dst.write(str(p) + "\n")
            append_file(dst, p)

        dst.write("\n" + "#" * 78 + "\n")
        dst.write("VERL SOURCE RANKING\n")
        dst.write("#" * 78 + "\n")
        for score, spath, _, counts in candidates[:80]:
            dst.write(
                f"score={score:4d} path={spath} counts="
                f"{json.dumps(counts, sort_keys=True)}\n"
            )

        for p in explicit + resolved:
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                append_file(dst, p)
                seen.add(key)

        included = 0
        for score, _, p, _ in candidates:
            if included >= MAX_SOURCE_FILES:
                break
            key = str(p.resolve())
            if key in seen:
                continue
            append_file(dst, p)
            seen.add(key)
            included += 1

        dst.write("\n" + "#" * 78 + "\n")
        dst.write("CANONICAL OPD PARQUET SCHEMA\n")
        dst.write("#" * 78 + "\n")
        dst.write(
            parquet_schema_text(
                ROOT / "data/verl_science_broad20k/opd_broad20k.parquet"
            )
        )

        dst.write("\n" + "#" * 78 + "\n")
        dst.write("PREPARE64 RUN FILE TREE\n")
        dst.write("#" * 78 + "\n")
        if run.is_dir():
            for p in sorted(run.rglob("*")):
                if p.is_file():
                    dst.write(
                        f"{p.relative_to(run)} bytes={p.stat().st_size}\n"
                    )
        else:
            dst.write("RUN_MISSING\n")

        dst.write("\n" + "#" * 78 + "\n")
        dst.write("ENV VERSIONS\n")
        dst.write("#" * 78 + "\n")
        dst.write(
            cmd(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys, torch, transformers; "
                        "print('python',sys.version); "
                        "print('torch',torch.__version__); "
                        "print('transformers',transformers.__version__); "
                        "import torch_npu; print('torch_npu',torch_npu.__version__)"
                    ),
                ]
            )
        )

    print(out)
    print(f"bytes={out.stat().st_size}")


if __name__ == "__main__":
    import sys
    main()
