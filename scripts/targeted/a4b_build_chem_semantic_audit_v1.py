#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/mtpatcher")

A4 = (
    ROOT
    / "runs/targeted"
    / "a4_kl_signal_localization_chemistry_v1_20260912"
)

OPD_AUDIT = (
    ROOT
    / "runs/targeted"
    / "wa_opd_trainset_audit_v1_20260911"
)

SFT_AUDIT = (
    ROOT
    / "runs/targeted"
    / "wa_sft_trainset_audit_c123_20260911"
)

H5_O2P5 = (
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
    / "a4b_chem_semantic_audit_v1_20260912"
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
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


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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

    os.replace(tmp, path)


def atomic_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open("w", encoding="utf-8") as f:
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

    os.replace(tmp, path)


def translation_of(row: dict[str, Any]) -> str | None:
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


def row_label(row: dict[str, Any]) -> str:
    for key in (
        "label",
        "arm",
        "model_label",
    ):
        x = row.get(key)

        if isinstance(x, str):
            return x.strip()

    return ""


def load_selected_meta():
    path = A4 / "rowwise_token_signals.jsonl"

    require(
        path.exists(),
        f"missing A4 rows: {path}",
    )

    rows = read_jsonl(path)

    require(
        len(rows) == 256,
        f"A4 row count={len(rows)} expected=256",
    )

    out = {}

    for r in rows:
        jid = int(r["job_id"])

        require(
            jid not in out,
            f"duplicate A4 job_id={jid}",
        )

        out[jid] = {
            "job_id": jid,
            "src_term": r["src_term"],
            "canonical_en_name": r["en_name"],
            "src_text": r["src_text"],
            "a4_c0_onpolicy_response":
                r["response_text"],
            "a4_canonical_hit":
                bool(r["canonical_en_name_hit"]),
        }

    return out, path


def compatible_row(
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

    src_term = row.get("src_term")

    if (
        isinstance(src_term, str)
        and src_term.strip()
        and src_term.strip() != meta["src_term"].strip()
    ):
        return False

    return True


def discover_label(
    root: Path,
    wanted_label: str,
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
    rows_seen = 0

    for path in sorted(root.rglob("*.jsonl")):
        files_seen += 1

        try:
            rows = read_jsonl(path)
        except Exception:
            continue

        for r in rows:
            rows_seen += 1

            if row_label(r) != wanted_label:
                continue

            jid_raw = r.get("job_id")

            if jid_raw is None:
                continue

            try:
                jid = int(jid_raw)
            except Exception:
                continue

            if jid not in selected:
                continue

            if not compatible_row(
                r,
                selected[jid],
            ):
                continue

            text = translation_of(r)

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
            ).append(str(path))

    return {
        "translations": found,
        "provenance": provenance,
        "conflicts": conflicts,
        "files_seen": files_seen,
        "rows_seen": rows_seen,
    }


def load_o2p5(
    selected: dict[int, dict[str, Any]],
):
    require(
        H5_O2P5.exists(),
        f"missing O2P5 file: {H5_O2P5}",
    )

    rows = read_jsonl(H5_O2P5)

    found = {}

    for r in rows:
        jid = int(r["job_id"])

        if jid not in selected:
            continue

        require(
            compatible_row(
                r,
                selected[jid],
            ),
            f"O2P5 metadata mismatch job={jid}",
        )

        text = translation_of(r)

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

    selected, a4_path = load_selected_meta()

    ids = list(selected)

    print(
        f"SELECTED_ROWS={len(ids)}"
    )

    print(
        "A4_ROW_SHA256="
        + sha256_file(a4_path)
    )

    c0 = discover_label(
        OPD_AUDIT,
        "C0",
        selected,
    )

    print(
        f"C0_DISCOVERED={len(c0['translations'])}/256 "
        f"files_seen={c0['files_seen']} "
        f"conflicts={len(c0['conflicts'])}"
    )

    require(
        not c0["conflicts"],
        "C0 conflicting translations found",
    )

    require(
        len(c0["translations"]) == 256,
        (
            "C0 incomplete "
            f"{len(c0['translations'])}/256"
        ),
    )

    o2p5 = load_o2p5(
        selected
    )

    print(
        f"O2P5_DISCOVERED={len(o2p5)}/256"
    )

    require(
        len(o2p5) == 256,
        (
            "O2P5 incomplete "
            f"{len(o2p5)}/256"
        ),
    )

    c2 = discover_label(
        SFT_AUDIT,
        "C2",
        selected,
    )

    print(
        f"C2_DISCOVERED={len(c2['translations'])}/256 "
        f"files_seen={c2['files_seen']} "
        f"conflicts={len(c2['conflicts'])}"
    )

    include_c2 = (
        len(c2["translations"]) == 256
        and not c2["conflicts"]
    )

    if include_c2:
        print("C2_INCLUDE=YES")
    else:
        print(
            "C2_INCLUDE=NO "
            "(semantic audit can proceed with C0/O2P5)"
        )

    systems = {
        "C0": c0["translations"],
        "O2P5": o2p5,
    }

    if include_c2:
        systems["C2"] = c2["translations"]

    audit_rows = []

    strict_summary = {}

    for label, translations in systems.items():
        hits = 0

        for jid in ids:
            m = selected[jid]
            hyp = translations[jid]

            hit = strict_hit(
                m["canonical_en_name"],
                hyp,
            )

            hits += int(hit)

            audit_rows.append(
                {
                    "eval_id":
                        f"{label}|chem-semantic|{jid}",
                    "label":
                        label,
                    "domain":
                        "chemistry",
                    "split":
                        "train",
                    "job_id":
                        jid,
                    "src_term":
                        m["src_term"],
                    "canonical_en_name":
                        m["canonical_en_name"],
                    "src_text":
                        m["src_text"],
                    "model_translation":
                        hyp,
                    "strict_canonical_hit":
                        hit,
                }
            )

        strict_summary[label] = {
            "n": 256,
            "hits": hits,
            "accuracy": hits / 256,
        }

    require(
        len(
            {
                r["eval_id"]
                for r in audit_rows
            }
        )
        == len(audit_rows),
        "duplicate eval_id",
    )

    out_path = (
        OUT
        / "chem_semantic_audit_input.jsonl"
    )

    atomic_jsonl(
        out_path,
        audit_rows,
    )

    manifest = {
        "status": "PASS",
        "scientific_class":
            "DIAGNOSTIC ONLY / NLP SEMANTIC LEXICAL AUDIT",
        "question": (
            "How much of the Chemistry strict-canonical gap "
            "reflects true entity translation error versus valid "
            "lexical aliases or registry nomenclature variation?"
        ),
        "selected_rows": 256,
        "systems": list(systems),
        "judge_rows": len(audit_rows),
        "strict_summary": strict_summary,
        "input_path": str(out_path),
        "input_sha256": sha256_file(out_path),
        "sources": {
            "A4_rows": str(a4_path),
            "C0_root": str(OPD_AUDIT),
            "O2P5_file": str(H5_O2P5),
            "C2_root": str(SFT_AUDIT),
        },
    }

    atomic_json(
        OUT / "manifest.json",
        manifest,
    )

    print()
    print("STRICT CANONICAL ON SAME 256")

    for label in systems:
        x = strict_summary[label]

        print(
            f"{label}: "
            f"{x['hits']}/{x['n']} "
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
        + sha256_file(out_path)
    )

    print(
        "FINAL_RESULT="
        "A4B_CHEM_SEMANTIC_AUDIT_PACK_PASS"
    )


if __name__ == "__main__":
    main()
