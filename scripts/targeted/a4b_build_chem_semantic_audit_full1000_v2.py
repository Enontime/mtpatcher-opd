#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os

from pathlib import Path
from typing import Any


ROOT = Path("/workspace/mtpatcher")

CONTEXT = (
    ROOT
    / "data/mtpatcher_v3_full6565_20260823"
    / "targeted_section43_contexts_qwen3_8b_final_20260910"
    / "chemistry_train.jsonl"
)

MANIFEST = (
    ROOT
    / "runs/targeted"
    / "wa_opd_horizon5_o12_20260911"
    / "learning_curve"
    / "train_matched"
    / "chemistry_sample_manifest.json"
)

C0_ROOT = (
    ROOT
    / "runs/targeted"
    / "wa_opd_trainset_audit_v1_20260911"
)

C2_ROOT = (
    ROOT
    / "runs/targeted"
    / "wa_sft_trainset_audit_c123_20260911"
)

O2P5_FILE = (
    ROOT
    / "runs/targeted"
    / "wa_opd_horizon5_o12_20260911"
    / "learning_curve"
    / "train_matched"
    / "O2"
    / "P5"
    / "train1000_translations.jsonl"
)

OUT = (
    ROOT
    / "runs/targeted"
    / "a4b_chem_semantic_audit_full1000_v2_20260912"
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception as e:
                raise RuntimeError(
                    f"{path}:{ln}: {e}"
                ) from e

    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def atomic_json(
    path: Path,
    obj: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp,
        path,
    )


def translation_of(
    row: dict[str, Any],
) -> str | None:
    for key in (
        "model_translation",
        "translation",
        "hypothesis",
        "prediction",
        "hyp",
    ):
        x = row.get(key)

        if isinstance(x, str) and x.strip():
            return x.strip()

    return None


def row_label(
    row: dict[str, Any],
) -> str:
    for key in (
        "label",
        "arm",
        "model_label",
    ):
        x = row.get(key)

        if isinstance(x, str):
            return x.strip()

    return ""


def load_matched1000():
    contexts = read_jsonl(
        CONTEXT
    )

    require(
        len(contexts) == 5500,
        f"context rows={len(contexts)} expected=5500",
    )

    by_id = {}

    for r in contexts:
        jid = int(r["job_id"])

        require(
            jid not in by_id,
            f"context duplicate job={jid}",
        )

        by_id[jid] = r

    manifest = json.loads(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    ids = [
        int(x)
        for x in manifest["job_ids"]
    ]

    require(
        len(ids) == 1000,
        f"manifest n={len(ids)}",
    )

    require(
        len(set(ids)) == 1000,
        "manifest duplicate job_id",
    )

    selected = {}

    for jid in ids:
        require(
            jid in by_id,
            f"manifest job missing context jid={jid}",
        )

        r = by_id[jid]

        require(
            r.get("domain") == "chemistry",
            f"wrong domain jid={jid}",
        )

        lr = r.get(
            "lexical_record"
        )

        require(
            isinstance(lr, dict),
            f"lexical_record missing jid={jid}",
        )

        en_name = str(
            lr.get(
                "en_name",
                "",
            )
        ).strip()

        require(
            en_name,
            f"en_name missing jid={jid}",
        )

        selected[jid] = {
            "job_id": jid,
            "src_term": r["src_term"],
            "canonical_en_name": en_name,
            "src_text": r["src_text"],
        }

    return ids, selected


def compatible(
    row: dict[str, Any],
    meta: dict[str, Any],
) -> bool:
    domain = row.get("domain")

    if (
        isinstance(domain, str)
        and domain.strip()
        and domain.strip() != "chemistry"
    ):
        return False

    split = row.get("split")

    if (
        isinstance(split, str)
        and split.strip()
        and split.strip() != "train"
    ):
        return False

    term = row.get(
        "src_term"
    )

    if (
        isinstance(term, str)
        and term.strip()
        and term.strip()
        != meta["src_term"].strip()
    ):
        return False

    src = row.get(
        "src_text"
    )

    if (
        isinstance(src, str)
        and src.strip()
        and src.strip()
        != meta["src_text"].strip()
    ):
        return False

    return True


def discover(
    root: Path,
    label: str,
    selected: dict[int, dict[str, Any]],
):
    require(
        root.exists(),
        f"root missing: {root}",
    )

    found: dict[int, str] = {}
    provenance: dict[int, list[str]] = {}

    conflicts = []

    files_seen = 0

    for path in sorted(
        root.rglob("*.jsonl")
    ):
        files_seen += 1

        try:
            rows = read_jsonl(
                path
            )
        except Exception:
            continue

        for row in rows:
            if row_label(row) != label:
                continue

            raw_jid = row.get(
                "job_id"
            )

            if raw_jid is None:
                continue

            try:
                jid = int(
                    raw_jid
                )
            except Exception:
                continue

            if jid not in selected:
                continue

            if not compatible(
                row,
                selected[jid],
            ):
                continue

            text = translation_of(
                row
            )

            if text is None:
                continue

            if jid in found:
                if found[jid] != text:
                    conflicts.append(
                        {
                            "job_id": jid,
                            "old": found[jid],
                            "new": text,
                            "path": str(path),
                        }
                    )

                    continue

            else:
                found[jid] = text

            provenance.setdefault(
                jid,
                [],
            ).append(
                str(path)
            )

    return {
        "translations":
            found,
        "provenance":
            provenance,
        "conflicts":
            conflicts,
        "files_seen":
            files_seen,
    }


def load_o2p5(
    selected: dict[int, dict[str, Any]],
):
    require(
        O2P5_FILE.exists(),
        f"missing O2P5 file: {O2P5_FILE}",
    )

    rows = read_jsonl(
        O2P5_FILE
    )

    found = {}

    for row in rows:
        raw_jid = row.get(
            "job_id"
        )

        if raw_jid is None:
            continue

        jid = int(
            raw_jid
        )

        if jid not in selected:
            continue

        require(
            compatible(
                row,
                selected[jid],
            ),
            f"O2P5 metadata mismatch job={jid}",
        )

        text = translation_of(
            row
        )

        require(
            text is not None,
            f"O2P5 translation missing job={jid}",
        )

        require(
            jid not in found,
            f"O2P5 duplicate job={jid}",
        )

        found[jid] = text

    return found


def strict_hit(
    canonical: str,
    translation: str,
) -> bool:
    return (
        canonical.casefold()
        in translation.casefold()
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ids, selected = (
        load_matched1000()
    )

    print(
        f"MATCHED_ROWS={len(ids)}"
    )

    c0 = discover(
        C0_ROOT,
        "C0",
        selected,
    )

    c2 = discover(
        C2_ROOT,
        "C2",
        selected,
    )

    o2p5 = load_o2p5(
        selected
    )

    print(
        "C0_DISCOVERED="
        f"{len(c0['translations'])}/1000 "
        f"files_seen={c0['files_seen']} "
        f"conflicts={len(c0['conflicts'])}"
    )

    print(
        "O2P5_DISCOVERED="
        f"{len(o2p5)}/1000"
    )

    print(
        "C2_DISCOVERED="
        f"{len(c2['translations'])}/1000 "
        f"files_seen={c2['files_seen']} "
        f"conflicts={len(c2['conflicts'])}"
    )

    require(
        not c0["conflicts"],
        "C0 conflicts found",
    )

    require(
        not c2["conflicts"],
        "C2 conflicts found",
    )

    require(
        len(c0["translations"])
        == 1000,
        (
            "C0 incomplete "
            f"{len(c0['translations'])}/1000"
        ),
    )

    require(
        len(o2p5)
        == 1000,
        (
            "O2P5 incomplete "
            f"{len(o2p5)}/1000"
        ),
    )

    require(
        len(c2["translations"])
        == 1000,
        (
            "C2 incomplete "
            f"{len(c2['translations'])}/1000"
        ),
    )

    systems = {
        "C0":
            c0["translations"],
        "O2P5":
            o2p5,
        "C2":
            c2["translations"],
    }

    strict_summary = {}
    audit_rows = []

    for label, translations in systems.items():
        hits = 0

        for jid in ids:
            meta = selected[jid]
            hyp = translations[jid]

            hit = strict_hit(
                meta[
                    "canonical_en_name"
                ],
                hyp,
            )

            hits += int(
                hit
            )

            audit_rows.append(
                {
                    "eval_id":
                        f"{label}|chem-semantic-full1000|{jid}",
                    "label":
                        label,
                    "domain":
                        "chemistry",
                    "split":
                        "train",
                    "job_id":
                        jid,
                    "src_term":
                        meta["src_term"],
                    "canonical_en_name":
                        meta[
                            "canonical_en_name"
                        ],
                    "src_text":
                        meta["src_text"],
                    "model_translation":
                        hyp,
                    "strict_canonical_hit":
                        hit,
                }
            )

        strict_summary[
            label
        ] = {
            "n": 1000,
            "hits": hits,
            "accuracy":
                hits / 1000,
        }

    require(
        strict_summary["C0"]["hits"]
        == 87,
        (
            "C0 strict provenance fail "
            f"got={strict_summary['C0']['hits']} "
            "expected=87"
        ),
    )

    require(
        strict_summary["O2P5"]["hits"]
        == 96,
        (
            "O2P5 strict provenance fail "
            f"got={strict_summary['O2P5']['hits']} "
            "expected=96"
        ),
    )

    require(
        len(audit_rows)
        == 3000,
        f"audit rows={len(audit_rows)}",
    )

    eval_ids = [
        r["eval_id"]
        for r in audit_rows
    ]

    require(
        len(set(eval_ids))
        == 3000,
        "duplicate eval_id",
    )

    out_path = (
        OUT
        / "chem_semantic_audit_full3000.jsonl"
    )

    atomic_jsonl(
        out_path,
        audit_rows,
    )

    manifest = {
        "status":
            "PASS",
        "scientific_class":
            "DIAGNOSTIC ONLY / NLP SEMANTIC LEXICAL AUDIT",
        "question":
            (
                "How much of the Chemistry canonical-string gap "
                "reflects true semantic entity translation error?"
            ),
        "paired_rows_per_system":
            1000,
        "systems":
            [
                "C0",
                "O2P5",
                "C2",
            ],
        "judge_rows":
            3000,
        "strict_summary":
            strict_summary,
        "input":
            str(out_path),
        "input_sha256":
            sha256_file(
                out_path
            ),
        "selection":
            {
                "manifest":
                    str(MANIFEST),
                "n":
                    1000,
                "paired":
                    True,
                "note":
                    (
                        "Uses the complete frozen matched Chemistry "
                        "train1000 manifest; no 256-row truncation."
                    ),
            },
    }

    atomic_json(
        OUT / "manifest.json",
        manifest,
    )

    print()
    print(
        "STRICT CANONICAL — FULL MATCHED1000"
    )

    for label in (
        "C0",
        "O2P5",
        "C2",
    ):
        x = strict_summary[
            label
        ]

        print(
            f"{label}: "
            f"{x['hits']}/1000 "
            f"accuracy={x['accuracy']:.6f}"
        )

    print()
    print(
        f"JUDGE_ROWS={len(audit_rows)}"
    )

    print(
        f"INPUT={out_path}"
    )

    print(
        "INPUT_SHA256="
        + sha256_file(
            out_path
        )
    )

    print(
        "FINAL_RESULT="
        "A4B_FULL1000_SEMANTIC_AUDIT_PACK_PASS"
    )


if __name__ == "__main__":
    main()
