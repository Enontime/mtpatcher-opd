#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path("/workspace/mtpatcher")
REPO = ROOT / "repo/MT-Patcher-Reproduction-Ascend"
PHASEA = ROOT / "runs/targeted/offline_prefix_support_replay_prepare64_v2_20260913"
PAIR = PHASEA / "paired/chemistry_pairs_first64.jsonl"
HASHES = PHASEA / "paired/chemistry_hashes_first64.json"
H5 = REPO / "scripts/targeted/targeted_wa_opd_horizon5_o12_v2.py"

STUDENT = ROOT / "models/Qwen3-0.6B"
TEACHER = ROOT / "models/Qwen3-8B"

EXPECTED_H5_SHA = "38ac62c0a93f1b762b7536db60421dd8c1e7aca49a57c25851306c3cb76b6e16"
EXPECTED_PAIR_SHA = "ac3c3975013393b9b75400f57f05a46273af5a9da0c9d705501990d1f47060d2"
EXPECTED_ROW_ORDER_SHA = "a13880e003fe1581dc0f945687a8d48533f957a4fce165229fd7d3b2f663d147"
EXPECTED_ROW_SET_SHA = "760f660aab2b2ef06cd600706a5fdf016588aeaaa135df78cfcc4bb996f95d21"
EXPECTED_ARM_SHA = {
    "S": "83231dd2da9eb018df0cc92de85a48a30c60ee01fab15149929555626481dccf",
    "T": "266a562b8a373446ede0dd7015dbcdbdbf3ae3f9dd3fc1e98de712b213f683e3",
}
EXPECTED_STUDENT_BANK_SHA = "5805dba61f4a87cd6d33b88a4ee26ad5de70322579f20be4626b2fec219f9b56"
EXPECTED_TEACHER_BANK_SHA = "21bd7570263496e8f9fd28aed9cb480e39eb1d5890b7b869925fb5d425531414"

DIRECT_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
DIRECT_PROMPT_SHA256 = "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"

TEACHER_CHEM_TEMPLATE = """You are translating a Chinese sentence into English.

Teacher-only lexical knowledge:
Chinese chemistry term: {src_term}
Canonical English registry term: {en_name}

Use the canonical English registry term when translating that designated
chemistry term. Translate the complete source sentence accurately and
naturally. Do not explain the instruction.

Chinese source:
{source}

Return only the English translation."""


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: {e}") from e
    return rows


def render_prompt_ids(tok, user_text: str) -> list[int]:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return [int(x) for x in tok(rendered, add_special_tokens=False)["input_ids"]]


def write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def build_arm(
    *,
    arm: str,
    pairs: list[dict[str, Any]],
    student_tok,
    teacher_tok,
    run: Path,
) -> dict[str, Any]:
    output_rows = []
    response_lists: list[list[int]] = []
    row_ids = []

    for index, pair in enumerate(pairs):
        require(pair["status"] == "PASS", f"pair status fail index={index}")
        require(pair["domain"] == "chemistry", f"domain drift index={index}")
        require(pair["row_id"] == f"chemistry:{pair['job_id']}", f"row_id drift index={index}")

        row_id = str(pair["row_id"])
        src_text = str(pair["src_text"])
        src_term = str(pair["src_term"])
        knowledge = str(pair["knowledge"])
        m_i = int(pair["m_i"])

        response_ids = [int(x) for x in pair[arm]["used_prefix_ids"]]
        response_sha = str(pair[arm]["used_prefix_sha256"])

        require(len(response_ids) == m_i, f"{row_id} m_i mismatch")
        require(sha256_json(response_ids) == response_sha, f"{row_id} response SHA mismatch")

        student_user = DIRECT_PROMPT.format(source=src_text)
        teacher_user = TEACHER_CHEM_TEMPLATE.format(
            src_term=src_term,
            en_name=knowledge,
            source=src_text,
        )
        student_prompt_ids = render_prompt_ids(student_tok, student_user)
        teacher_prompt_ids = render_prompt_ids(teacher_tok, teacher_user)

        require(knowledge in teacher_user, f"{row_id} Teacher hint absent")
        require(student_user == DIRECT_PROMPT.format(source=src_text), f"{row_id} Student prompt drift")

        extra_info = {
            "index": index,
            "row_id": row_id,
            "job_id": int(pair["job_id"]),
            "domain": "chemistry",
            "arm": arm,
            "src_text": src_text,
            "src_term": src_term,
            "knowledge": knowledge,
            "m_i": m_i,
            "frozen_response_ids": response_ids,
            "frozen_response_sha256": response_sha,
            "student_user_text": student_user,
            "teacher_user_text": teacher_user,
            "student_prompt_ids": student_prompt_ids,
            "teacher_prompt_ids": teacher_prompt_ids,
            "student_prompt_sha256": sha256_json(student_prompt_ids),
            "teacher_prompt_sha256": sha256_json(teacher_prompt_ids),
            "teacher_hint_present": True,
            "student_hint_present": False,
            "formal_training_authorized": False,
        }

        output_rows.append(
            {
                "data_source": "default",
                "prompt": [{"role": "user", "content": student_user}],
                "ability": "translation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "UNUSED_ZERO_REWARD",
                },
                "extra_info": extra_info,
            }
        )
        response_lists.append(response_ids)
        row_ids.append(row_id)

    require(len(output_rows) == 64, f"{arm} rows={len(output_rows)}")
    aggregate_sha = sha256_json(response_lists)
    require(
        aggregate_sha == EXPECTED_ARM_SHA[arm],
        f"{arm} aggregate response SHA mismatch got={aggregate_sha}",
    )
    require(sha256_json(row_ids) == EXPECTED_ROW_ORDER_SHA, f"{arm} row-order SHA drift")

    # Partition-invariance gate: serialization of the ordered frozen IDs must not
    # depend on how a later trainer chunks the 64 examples.
    partition_hashes = {}
    for part in (1, 2, 4, 8, 16, 32, 64):
        reconstructed = []
        for begin in range(0, len(response_lists), part):
            reconstructed.extend(response_lists[begin : begin + part])
        h = sha256_json(reconstructed)
        require(h == aggregate_sha, f"{arm} partition SHA drift part={part}")
        partition_hashes[str(part)] = h

    path = run / "inputs" / f"chemistry_{arm}_first64.parquet"
    write_parquet_atomic(path, output_rows)

    # Parse back through Arrow immediately; this catches nested-schema transport
    # surprises before a model process is launched.
    back = pq.read_table(path).to_pylist()
    require(len(back) == 64, f"{arm} parquet row count after readback")
    back_ids = [
        [int(x) for x in row["extra_info"]["frozen_response_ids"]]
        for row in back
    ]
    require(sha256_json(back_ids) == aggregate_sha, f"{arm} parquet response IDs changed")

    return {
        "arm": arm,
        "rows": 64,
        "parquet": str(path),
        "parquet_sha256": sha256_file(path),
        "aggregate_response_ids_sha256": aggregate_sha,
        "partition_invariance_sha256": partition_hashes,
        "row_order_sha256": sha256_json(row_ids),
        "student_hint_present": False,
        "teacher_hint_present": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)

    require(PAIR.is_file(), f"missing pair file: {PAIR}")
    require(HASHES.is_file(), f"missing hashes file: {HASHES}")
    require(sha256_file(PAIR) == EXPECTED_PAIR_SHA, "chemistry pair file SHA drift")
    require(sha256_file(H5) == EXPECTED_H5_SHA, "H5 source SHA drift")
    require(
        hashlib.sha256(DIRECT_PROMPT.encode("utf-8")).hexdigest() == DIRECT_PROMPT_SHA256,
        "direct prompt SHA drift",
    )

    frozen = json.loads(HASHES.read_text(encoding="utf-8"))
    require(frozen["rows"] == 64, "frozen hash row count drift")
    require(frozen["row_order_sha256"] == EXPECTED_ROW_ORDER_SHA, "row-order hash drift")
    require(frozen["row_set_sha256"] == EXPECTED_ROW_SET_SHA, "row-set hash drift")
    require(
        frozen["matched_S_used_ids_sha256"] == EXPECTED_ARM_SHA["S"],
        "frozen matched S hash drift",
    )
    require(
        frozen["matched_T_used_ids_sha256"] == EXPECTED_ARM_SHA["T"],
        "frozen matched T hash drift",
    )
    require(
        frozen["student_prefix_bank_sha256"] == EXPECTED_STUDENT_BANK_SHA,
        "Student bank hash drift",
    )
    require(
        frozen["teacher_prefix_bank_sha256"] == EXPECTED_TEACHER_BANK_SHA,
        "Teacher bank hash drift",
    )

    pairs = read_jsonl(PAIR)
    require(len(pairs) == 64, f"chemistry pair rows={len(pairs)}")

    student_tok = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    require(student_tok.pad_token_id is not None, "Student pad token missing")
    require(teacher_tok.pad_token_id is not None, "Teacher pad token missing")
    require(student_tok.eos_token_id == teacher_tok.eos_token_id, "Student/Teacher EOS mismatch")

    arms = {}
    for arm in ("S", "T"):
        arms[arm] = build_arm(
            arm=arm,
            pairs=pairs,
            student_tok=student_tok,
            teacher_tok=teacher_tok,
            run=run,
        )

    manifest = {
        "status": "PASS_INPUTS_FROZEN_FOR_ENGINEERING_SMOKE",
        "scientific_class": "DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY",
        "formal_training_authorized": False,
        "parameter_updates_per_arm": 1,
        "parameter_update_semantics": "disposable engineering smoke only",
        "source_pair_file": str(PAIR),
        "source_pair_sha256": EXPECTED_PAIR_SHA,
        "row_order_sha256": EXPECTED_ROW_ORDER_SHA,
        "row_set_sha256": EXPECTED_ROW_SET_SHA,
        "student_prefix_bank_sha256": EXPECTED_STUDENT_BANK_SHA,
        "teacher_prefix_bank_sha256": EXPECTED_TEACHER_BANK_SHA,
        "matched_response_ids_sha256": EXPECTED_ARM_SHA,
        "direct_prompt_sha256": DIRECT_PROMPT_SHA256,
        "h5_source_sha256": EXPECTED_H5_SHA,
        "rows": 64,
        "domain": "chemistry",
        "arms": arms,
    }
    atomic_json(run / "input_manifest.json", manifest)

    print("SMOKE64_INPUT_BUILD=PASS")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
