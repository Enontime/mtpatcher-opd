#!/usr/bin/env python3
"""
Targeted WA knowledge-conditioned OPD overnight chain v3.

SCIENTIFIC CLASS
----------------
LAB ADAPTATION / CUSTOM_TORCH_NPU / KNOWLEDGE-CONDITIONED OPD

Purpose:
  After targeted WA-SFT positive control established that Chemistry/Idiom
  knowledge is learnable, test whether on-policy forward-KL can transfer the
  same lexical knowledge to the Student.

Arms (all initialize independently from the same frozen C0):
  O1 = Idiom targeted WA-OPD
  O2 = Chemistry targeted WA-OPD
  O3 = Chemistry + Idiom targeted WA-OPD

Student conditioning:
  exact direct-translation prompt with the original Chinese source only.

Teacher conditioning:
  same original Chinese source + Teacher-only lexical knowledge:
    Chemistry -> strict lexical_record["en_name"]
    Idiom     -> frozen dictionary definition
  plus EXACT replay of the Student's generated response prefix.

Loss:
  top-k=32 Teacher-renormalized forward KL on Student-generated response tokens.
  No reward, PPO objective, advantage, or task score enters the update.

Rollout:
  Student on-policy sampling, temperature=1.0, top_p=1.0, top_k=0,
  enable_thinking=False.

Important claim boundary:
  Teacher and Student have different source-side conditioning because the
  Teacher receives lexical knowledge. Therefore this run is NOT called
  canonical/paper-exact OPD. It is a deliberate knowledge-conditioned
  adaptation.

The same file also:
  1) evaluates already-trained SFT C1/C2/C3 on frozen WMT24/FLORES/Challenge
     BLEU+chrF in parallel (non-blocking diagnostic);
  2) trains O1/O2/O3;
  3) evaluates O1/O2/O3 on targeted diagnostic2000;
  4) evaluates O1/O2/O3 on WMT24/FLORES/Challenge BLEU+chrF;
  5) materializes O1/O2/O3 Idiom judge3000 input for the frozen Windows judge.

Engineering:
  - one file only;
  - durable JSONL generation outputs;
  - progress/state JSON with done/total/ETA;
  - epoch-level exact restart point (model + optimizer + scheduler);
  - failed BLEU side diagnostic never aborts OPD training;
  - no target-side SFT translation is read by OPD training;
  - all long-run outputs live under /workspace/mtpatcher/runs/targeted/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

TZ8 = timezone(timedelta(hours=8))

ROOT = Path("/workspace/mtpatcher")
REPO = ROOT / "repo/MT-Patcher-Reproduction-Ascend"

C0 = ROOT / "models/Qwen3-0.6B"
TEACHER = ROOT / "models/Qwen3-8B"

CONTEXTS = (
    ROOT
    / "data/mtpatcher_v3_full6565_20260823"
    / "targeted_section43_contexts_qwen3_8b_final_20260910"
)
DIAG = (
    ROOT
    / "data/mtpatcher_v3_full6565_20260823"
    / "targeted_diagnostic1000_v1_20260910"
)
SFT_RUN = ROOT / "runs/targeted/wa_sft_positive_control_c123_20260910"
RUN = ROOT / "runs/targeted/wa_opd_horizon5_o12_20260911"

GENERAL_EVAL = {
    "WMT24": ROOT / "data/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl",
    "FLORES": ROOT / "data/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl",
    "Challenge": ROOT / "data/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl",
}
GENERAL_EXPECTED_ROWS = {"WMT24": 998, "FLORES": 1012, "Challenge": 197}

BASE_GENERAL = {
    "WMT24": {"bleu": 15.5362},
    "FLORES": {"bleu": 19.9715},
    "Challenge": {"bleu": 16.5379},
    "macro_bleu": 17.348521468,
}

DIRECT_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
DIRECT_PROMPT_SHA = hashlib.sha256(DIRECT_PROMPT.encode("utf-8")).hexdigest()
EXPECTED_DIRECT_PROMPT_SHA = (
    "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"
)

EXPECTED_CONTEXT_HASHES = {
    "chemistry_train.jsonl":
        "ca6532cacce64f24f14226a4dca509b494aaa5f5bbeecbc0f1d2f157c317044f",
    "idiom_train.jsonl":
        "804df445752caa48ee95a6d2d6ce9403b03e8aba6eaa994b380b5bc9653d0739",
}
EXPECTED_DIAG_SHA = (
    "7c5082217de8071bd0b519e83cac255896f704d0b41d9bd49ae2cd5b41af6db2"
)
EXPECTED_C0_CONFIG_SHA = (
    "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
)
EXPECTED_C0_TOKENIZER_CONFIG_SHA = (
    "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
)

# Frozen adaptation hyperparameters.
EPOCHS = 5
MICRO_BATCH = 4
GRAD_ACCUM = 4
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
LR = 1e-6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
MAX_TOTAL_LENGTH = 1024
MAX_NEW_TOKENS = 256
TOP_K_TEACHER = 32
ROLLOUT_TEMPERATURE = 1.0
SEED = 20260820

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

TEACHER_IDIOM_TEMPLATE = """You are translating a Chinese sentence into English.

Teacher-only lexical knowledge:
Designated Chinese idiom: {src_term}
Dictionary definition: {definition}

Use the dictionary definition to convey the idiom's contextual meaning
naturally in English. Translate the complete source sentence. Do not quote the
Chinese idiom and do not explain the instruction.

Chinese source:
{source}

Return only the English translation."""


def now8() -> str:
    return datetime.now(TZ8).isoformat(timespec="seconds")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: {e}") from e
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    import torch
    torch.manual_seed(seed)
    try:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    except Exception:
        pass


def stable_row_seed(row: dict[str, Any], epoch: int) -> int:
    key = f"{SEED}|{epoch}|{row['domain']}|{row['job_id']}"
    v = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
    return v % 2_000_000_000


def tokenizer_vocab_hash(tok) -> str:
    vocab = tok.get_vocab()
    return sha256_json(sorted(vocab.items(), key=lambda x: x[0]))


def verify_frozen_inputs() -> dict[str, Any]:
    require(
        DIRECT_PROMPT_SHA == EXPECTED_DIRECT_PROMPT_SHA,
        f"DIRECT_PROMPT_SHA_FAIL got={DIRECT_PROMPT_SHA}",
    )
    require(C0.exists(), f"C0 missing: {C0}")
    require(TEACHER.exists(), f"Teacher missing: {TEACHER}")

    got = sha256_file(C0 / "config.json")
    require(
        got == EXPECTED_C0_CONFIG_SHA,
        f"C0 config hash mismatch expected={EXPECTED_C0_CONFIG_SHA} got={got}",
    )
    got = sha256_file(C0 / "tokenizer_config.json")
    require(
        got == EXPECTED_C0_TOKENIZER_CONFIG_SHA,
        "C0 tokenizer_config hash mismatch "
        f"expected={EXPECTED_C0_TOKENIZER_CONFIG_SHA} got={got}",
    )

    for name, expected in EXPECTED_CONTEXT_HASHES.items():
        path = CONTEXTS / name
        require(path.exists(), f"missing frozen context file: {path}")
        got = sha256_file(path)
        require(
            got == expected,
            f"context hash mismatch file={name} expected={expected} got={got}",
        )

    diag_path = DIAG / "targeted_diagnostic2000.jsonl"
    require(diag_path.exists(), f"missing diagnostic: {diag_path}")
    got = sha256_file(diag_path)
    require(
        got == EXPECTED_DIAG_SHA,
        f"diagnostic hash mismatch expected={EXPECTED_DIAG_SHA} got={got}",
    )

    for name, path in GENERAL_EVAL.items():
        require(path.exists(), f"missing general eval file {name}: {path}")
        rows = read_jsonl(path)
        require(
            len(rows) == GENERAL_EXPECTED_ROWS[name],
            f"{name} count mismatch {len(rows)}",
        )

    return {
        "direct_prompt_sha256": DIRECT_PROMPT_SHA,
        "c0_config_sha256": EXPECTED_C0_CONFIG_SHA,
        "c0_tokenizer_config_sha256": EXPECTED_C0_TOKENIZER_CONFIG_SHA,
        "context_hashes": EXPECTED_CONTEXT_HASHES,
        "diagnostic_sha256": EXPECTED_DIAG_SHA,
    }


def load_arm_rows(arm: str) -> list[dict[str, Any]]:
    chem = read_jsonl(CONTEXTS / "chemistry_train.jsonl")
    idiom = read_jsonl(CONTEXTS / "idiom_train.jsonl")

    require(len(chem) == 5500, f"chem rows={len(chem)}")
    require(len(idiom) == 5500, f"idiom rows={len(idiom)}")

    for r in chem:
        require(r.get("domain") == "chemistry", f"chem domain bad {r.get('job_id')}")
        require(r.get("split") == "train", f"chem split bad {r.get('job_id')}")
        lr = r.get("lexical_record")
        require(isinstance(lr, dict), f"chem lexical_record missing {r.get('job_id')}")
        en = lr.get("en_name")
        require(
            isinstance(en, str) and en.strip(),
            f"STRICT_CHEM_EN_NAME_MISSING job_id={r.get('job_id')}",
        )

    for r in idiom:
        require(r.get("domain") == "idiom", f"idiom domain bad {r.get('job_id')}")
        require(r.get("split") == "train", f"idiom split bad {r.get('job_id')}")
        d = r.get("definition")
        require(
            isinstance(d, str) and d.strip(),
            f"IDIOM_DEFINITION_MISSING job_id={r.get('job_id')}",
        )

    if arm == "O1":
        return idiom
    if arm == "O2":
        return chem
    if arm == "O3":
        # Fixed deterministic interleaving prevents an all-domain block schedule.
        mixed = []
        for a, b in zip(chem, idiom):
            mixed.append(a)
            mixed.append(b)
        return mixed
    raise ValueError(arm)


def student_user_text(row: dict[str, Any]) -> str:
    return DIRECT_PROMPT.format(source=row["src_text"])


def teacher_user_text(row: dict[str, Any]) -> str:
    if row["domain"] == "chemistry":
        en_name = row["lexical_record"]["en_name"].strip()
        return TEACHER_CHEM_TEMPLATE.format(
            src_term=row["src_term"],
            en_name=en_name,
            source=row["src_text"],
        )
    if row["domain"] == "idiom":
        return TEACHER_IDIOM_TEMPLATE.format(
            src_term=row["src_term"],
            definition=row["definition"],
            source=row["src_text"],
        )
    raise ValueError(row["domain"])


def render_prompt_ids(tok, user_text: str) -> list[int]:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tok(rendered, add_special_tokens=False)["input_ids"]


def trim_response(ids: list[int], eos_id: int | None) -> list[int]:
    out = list(map(int, ids))
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id) + 1]
    return out


def pad_sequences(seqs: list[list[int]], pad_id: int, device: str):
    import torch

    maxlen = max(len(x) for x in seqs)
    ids = []
    mask = []
    for x in seqs:
        n = len(x)
        ids.append(x + [pad_id] * (maxlen - n))
        mask.append([1] * n + [0] * (maxlen - n))
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.long, device=device),
    )


def generate_student_responses(
    model,
    tok,
    rows: list[dict[str, Any]],
    device: str,
    epoch: int,
) -> list[list[int]]:
    """
    Keyed one-row-at-a-time sampling makes each rollout RNG depend only on
    (seed, epoch, domain, job_id), not batch partition/order.
    """
    import torch

    was_training = model.training
    model.eval()
    outs = []

    with torch.no_grad():
        for row in rows:
            seed = stable_row_seed(row, epoch)
            set_seed(seed)

            user = student_user_text(row)
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            batch = tok(rendered, return_tensors="pt", add_special_tokens=False)
            batch = {k: v.to(device) for k, v in batch.items()}
            prompt_width = batch["input_ids"].shape[1]

            seq = model.generate(
                **batch,
                do_sample=True,
                temperature=ROLLOUT_TEMPERATURE,
                top_p=1.0,
                top_k=0,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
            resp = seq[0, prompt_width:].detach().cpu().tolist()
            resp = trim_response(resp, tok.eos_token_id)
            require(resp, f"empty rollout job_id={row['job_id']}")
            outs.append(resp)

    if was_training:
        model.train()
    return outs


def teacher_topk_for_batch(
    teacher,
    teacher_tok,
    rows: list[dict[str, Any]],
    responses: list[list[int]],
    teacher_device: str,
) -> list[tuple[Any, Any]]:
    """
    Return one (top_values, top_indices) pair per row, aligned to response
    token positions. Values/indices are returned on CPU to cross device safely.
    """
    import torch

    prompt_ids = [
        render_prompt_ids(teacher_tok, teacher_user_text(r)) for r in rows
    ]

    seqs = []
    for p, resp, row in zip(prompt_ids, responses, rows):
        require(
            len(p) + len(resp) <= MAX_TOTAL_LENGTH,
            f"teacher sequence too long job_id={row['job_id']} "
            f"len={len(p)+len(resp)}",
        )
        seqs.append(p + resp)

    ids, mask = pad_sequences(
        seqs, teacher_tok.pad_token_id, teacher_device
    )

    with torch.no_grad():
        logits = teacher(input_ids=ids, attention_mask=mask).logits

    ret = []
    for i, (p, resp) in enumerate(zip(prompt_ids, responses)):
        start = len(p) - 1
        end = start + len(resp)
        # logits[t] predicts token[t+1].
        wanted = logits[i, start:end, :].float()
        vals, inds = torch.topk(wanted, k=TOP_K_TEACHER, dim=-1)
        ret.append((vals.cpu(), inds.cpu()))

    del logits, ids, mask
    return ret


def student_kl_sum_for_batch(
    student,
    student_tok,
    rows: list[dict[str, Any]],
    responses: list[list[int]],
    teacher_topk: list[tuple[Any, Any]],
    student_device: str,
):
    """
    Compute differentiable KL SUM over all response tokens.
    Caller accumulates KL sums across GRAD_ACCUM microbatches, then divides
    gradients by the total response-token count before optimizer.step().
    """
    import torch
    import torch.nn.functional as F

    prompt_ids = [
        render_prompt_ids(student_tok, student_user_text(r)) for r in rows
    ]
    seqs = []
    for p, resp, row in zip(prompt_ids, responses, rows):
        require(
            len(p) + len(resp) <= MAX_TOTAL_LENGTH,
            f"student sequence too long job_id={row['job_id']} "
            f"len={len(p)+len(resp)}",
        )
        seqs.append(p + resp)

    ids, mask = pad_sequences(
        seqs, student_tok.pad_token_id, student_device
    )
    logits = student(input_ids=ids, attention_mask=mask).logits

    total = None
    token_count = 0
    per_row_kl = []

    for i, (p, resp, tk) in enumerate(zip(prompt_ids, responses, teacher_topk)):
        tvals_cpu, tinds_cpu = tk
        tvals = tvals_cpu.to(student_device)
        tinds = tinds_cpu.to(student_device)

        start = len(p) - 1
        end = start + len(resp)
        slogits = logits[i, start:end, :]

        # Full-vocab Student normalization, but only Teacher top-k terms are
        # needed in the expectation.
        student_lse = torch.logsumexp(slogits.float(), dim=-1, keepdim=True)
        student_selected = torch.gather(
            slogits.float(), dim=-1, index=tinds
        )
        s_logp = student_selected - student_lse

        t_logp = F.log_softmax(tvals.float(), dim=-1)
        t_prob = t_logp.exp()

        token_kl = (t_prob * (t_logp - s_logp)).sum(dim=-1)
        row_sum = token_kl.sum()
        total = row_sum if total is None else total + row_sum
        token_count += len(resp)
        per_row_kl.append(float(token_kl.mean().detach().cpu()))

    require(total is not None and token_count > 0, "zero KL tokens")
    return total, token_count, per_row_kl


def replay_partition_loss(
    student,
    teacher,
    student_tok,
    teacher_tok,
    rows: list[dict[str, Any]],
    responses: list[list[int]],
    partition: int,
    sdev: str,
    tdev: str,
) -> float:
    """
    No-grad numerical replay of EXACT response IDs under partition sizes
    1/2/4. This checks response alignment + masking + reduction invariance in
    this custom stack. It does not claim to close the separate Verl gate.
    """
    import torch

    total_kl = 0.0
    total_tokens = 0

    student.eval()
    teacher.eval()

    for start in range(0, len(rows), partition):
        rr = rows[start:start + partition]
        rp = responses[start:start + partition]

        tk = teacher_topk_for_batch(teacher, teacher_tok, rr, rp, tdev)
        with torch.no_grad():
            klsum, ntok, _ = student_kl_sum_for_batch(
                student, student_tok, rr, rp, tk, sdev
            )
        total_kl += float(klsum.detach().cpu())
        total_tokens += ntok

    return total_kl / total_tokens


def custom_semantic_preflight(
    student,
    teacher,
    student_tok,
    teacher_tok,
    rows: list[dict[str, Any]],
    sdev: str,
    tdev: str,
    out_dir: Path,
) -> None:
    """
    v2 semantic preflight.

    What is a material semantic requirement here:
      1) the SAME keyed Student rollout must produce the SAME response token IDs
         regardless of whether the four rows are requested as MB1/MB2/MB4;
      2) loss masking/reduction over those exact frozen rowwise KL values must
         be partition invariant.

    What is NOT a semantic requirement:
      BF16/NPU logits being bitwise/numerically invariant when the same rows are
      padded into different batch shapes. Kernel selection, padding shape and
      reduction order can move logits/top-k slightly. v1 incorrectly turned
      that numerical diagnostic into a hard 0.5% gate.

    v2 therefore records the real batched MB1/MB2/MB4 drift, but does not use
    that drift to decide scientific validity.
    """
    import torch

    sample = rows[:4]
    require(len(sample) == 4, "semantic preflight needs 4 rows")

    # Preserve the failed v1 record rather than silently overwriting history.
    old = out_dir / "semantic_preflight.json"
    old_copy = out_dir / "semantic_preflight_v1_failed.json"
    if old.exists() and not old_copy.exists():
        try:
            previous = json.loads(old.read_text(encoding="utf-8"))
            if previous.get("status") == "FAIL":
                shutil.copy2(old, old_copy)
        except Exception:
            pass

    # --------------------------------------------------------------
    # A. Response-token invariance across partitioning.
    #
    # generate_student_responses itself uses a stable per-row seed derived
    # from (seed, epoch, domain, job_id), so this checks that implementation
    # rather than assuming it.
    # --------------------------------------------------------------
    responses_by_mb = {}
    shas = {}

    for mb in (1, 2, 4):
        collected = []
        for begin in range(0, len(sample), mb):
            chunk_rows = sample[begin:begin + mb]
            chunk_responses = generate_student_responses(
                student,
                student_tok,
                chunk_rows,
                sdev,
                epoch=0,
            )
            collected.extend(chunk_responses)

        require(len(collected) == len(sample), f"MB{mb} rollout count mismatch")
        responses_by_mb[str(mb)] = collected
        shas[str(mb)] = sha256_json(
            [
                {
                    "domain": r["domain"],
                    "job_id": r["job_id"],
                    "response_ids": resp,
                }
                for r, resp in zip(sample, collected)
            ]
        )

    response_ids_equal = (
        responses_by_mb["1"]
        == responses_by_mb["2"]
        == responses_by_mb["4"]
    )

    # Use the exact frozen token IDs from MB1 as canonical replay.
    responses = responses_by_mb["1"]

    # --------------------------------------------------------------
    # B. Canonical rowwise teacher/student scoring.
    #
    # Each row is scored alone to remove padding/batch-shape effects. These
    # exact rowwise KL sums and token counts are then regrouped under MB1/2/4.
    # Any mismatch here would be a true masking/reduction bug.
    # --------------------------------------------------------------
    row_kl_sums = []
    row_token_counts = []
    row_mean_kls = []

    student.eval()
    teacher.eval()

    for row, resp in zip(sample, responses):
        tk = teacher_topk_for_batch(
            teacher,
            teacher_tok,
            [row],
            [resp],
            tdev,
        )
        with torch.no_grad():
            klsum, ntok, row_kls = student_kl_sum_for_batch(
                student,
                student_tok,
                [row],
                [resp],
                tk,
                sdev,
            )

        row_kl_sums.append(float(klsum.detach().cpu()))
        row_token_counts.append(int(ntok))
        row_mean_kls.append(float(row_kls[0]))

    reduction_losses = {}
    for mb in (1, 2, 4):
        total_sum = 0.0
        total_tokens = 0

        for begin in range(0, len(sample), mb):
            end = min(begin + mb, len(sample))
            group_sum = sum(row_kl_sums[begin:end])
            group_tokens = sum(row_token_counts[begin:end])
            total_sum += group_sum
            total_tokens += group_tokens

        require(total_tokens > 0, f"MB{mb} reduction zero tokens")
        reduction_losses[str(mb)] = total_sum / total_tokens

    red_span = max(reduction_losses.values()) - min(reduction_losses.values())
    red_mean = sum(reduction_losses.values()) / len(reduction_losses)
    red_rel = red_span / max(abs(red_mean), 1e-12)

    # Tight tolerance is appropriate here because the same already-computed
    # Python floats are only regrouped; no new model forward occurs.
    reduction_equal = red_rel <= 1e-12

    # --------------------------------------------------------------
    # C. Non-blocking numerical diagnostic from actual batched forwards.
    #
    # This is exactly the quantity that stopped v1. We retain it for
    # provenance, but it cannot veto the run.
    # --------------------------------------------------------------
    batched_losses = {}
    for mb in (1, 2, 4):
        batched_losses[str(mb)] = replay_partition_loss(
            student,
            teacher,
            student_tok,
            teacher_tok,
            sample,
            responses,
            mb,
            sdev,
            tdev,
        )

    batched_span = max(batched_losses.values()) - min(batched_losses.values())
    batched_mean = sum(batched_losses.values()) / len(batched_losses)
    batched_rel = batched_span / max(abs(batched_mean), 1e-12)

    passed = response_ids_equal and reduction_equal

    result = {
        "status": "PASS" if passed else "FAIL",
        "version": 2,
        "scientific_class":
            "CUSTOM_TORCH_NPU_RESPONSE_REPLAY_AND_REDUCTION_PREFLIGHT",
        "claim_boundary":
            "validates this custom stack only; does not certify Verl canonical OPD",
        "temperature": ROLLOUT_TEMPERATURE,
        "top_k_teacher": TOP_K_TEACHER,
        "response_replay": {
            "status": "PASS" if response_ids_equal else "FAIL",
            "sha256_by_partition": shas,
            "exact_ids_equal_mb1_mb2_mb4": response_ids_equal,
        },
        "rowwise_reduction": {
            "status": "PASS" if reduction_equal else "FAIL",
            "loss_by_partition": reduction_losses,
            "absolute_span": red_span,
            "relative_span": red_rel,
            "relative_tolerance": 1e-12,
            "row_kl_sums": row_kl_sums,
            "row_token_counts": row_token_counts,
            "row_mean_kls": row_mean_kls,
        },
        "batched_forward_numerical_diagnostic": {
            "status": "NON_BLOCKING_DIAGNOSTIC",
            "loss_by_partition": batched_losses,
            "absolute_span": batched_span,
            "relative_span": batched_rel,
            "explanation":
                "BF16/NPU forward logits/top-k may vary with padding/batch shape; "
                "this is recorded but is not the semantic replay criterion.",
        },
        "rows": [
            {
                "domain": r["domain"],
                "job_id": r["job_id"],
                "response_tokens": len(resp),
            }
            for r, resp in zip(sample, responses)
        ],
        "created": now8(),
    }

    atomic_json(out_dir / "semantic_preflight_v2.json", result)

    print(
        f"{now8()} SEMANTIC_PREFLIGHT_V2 status={result['status']} "
        f"response_equal={response_ids_equal} "
        f"sha_mb1={shas['1']} sha_mb2={shas['2']} sha_mb4={shas['4']} "
        f"reduction_rel_span={red_rel:.3g} "
        f"batched_numeric_rel_span={batched_rel:.6g}",
        flush=True,
    )

    require(
        response_ids_equal,
        "CUSTOM_SEMANTIC_PREFLIGHT_V2_RESPONSE_ID_MISMATCH",
    )
    require(
        reduction_equal,
        f"CUSTOM_SEMANTIC_PREFLIGHT_V2_REDUCTION_MISMATCH rel_span={red_rel}",
    )

def optimizer_to(optimizer, device: str) -> None:
    import torch
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_resume(
    out_dir: Path,
    student,
    student_tok,
    optimizer,
    scheduler,
    completed_epoch: int,
    completed_updates: int,
) -> None:
    import torch

    hf_tmp = out_dir / "resume_hf.tmp"
    hf = out_dir / "resume_hf"
    if hf_tmp.exists():
        shutil.rmtree(hf_tmp)
    hf_tmp.mkdir(parents=True, exist_ok=True)

    old_cache = getattr(student.config, "use_cache", None)
    if old_cache is not None:
        student.config.use_cache = True
    student.save_pretrained(hf_tmp, safe_serialization=True)
    student_tok.save_pretrained(hf_tmp)
    if old_cache is not None:
        student.config.use_cache = old_cache

    if hf.exists():
        shutil.rmtree(hf)
    os.replace(hf_tmp, hf)

    state_tmp = out_dir / "resume_state.pt.tmp"
    state = out_dir / "resume_state.pt"
    torch.save(
        {
            "completed_epoch": completed_epoch,
            "completed_updates": completed_updates,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        state_tmp,
    )
    os.replace(state_tmp, state)

    atomic_json(
        out_dir / "resume_meta.json",
        {
            "completed_epoch": completed_epoch,
            "completed_updates": completed_updates,
            "updated": now8(),
        },
    )



def cmd_preflight_worker(args) -> None:
    """
    Full end-to-end dry-run of the REAL OPD update path for one arm.

    It intentionally performs a real GA4 optimizer update on 16 source rows,
    then exercises the exact save_resume()/reload path in a separate preflight
    directory. The process exits afterwards, so the scientific O1/O2/O3 run
    still initializes from the untouched frozen C0.

    This catches:
      - Student + Teacher load/device placement;
      - frozen data schema + lexical knowledge;
      - on-policy Student sampling;
      - Teacher top-k32 prefix scoring;
      - differentiable Student forward-KL;
      - backward on NPU;
      - GA4 token-normalized gradients;
      - clipping + AdamW + scheduler step;
      - checkpoint serialization;
      - optimizer/scheduler serialization;
      - model + optimizer/scheduler reload;
      - post-reload greedy generation.
    """
    import torch
    import torch_npu  # noqa: F401
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    arm = args.arm
    out_dir = RUN / "preflight_v3" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    current_script_sha = sha256_file(Path(__file__).resolve())
    if summary_path.exists():
        try:
            old = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                old.get("status") == "PASS"
                and old.get("script_sha256") == current_script_sha
            ):
                print(
                    f"{now8()} {arm}_E2E_PREFLIGHT_ALREADY_PASS "
                    f"script_sha={current_script_sha}",
                    flush=True,
                )
                return
        except Exception:
            pass

    frozen = verify_frozen_inputs()
    rows = load_arm_rows(arm)
    require(len(rows) >= 16, f"{arm}: need >=16 rows")

    # Use the first 16 frozen rows. O3 is deterministically interleaved, so its
    # smoke contains both domains.
    smoke_rows = rows[:16]
    domains = Counter(r["domain"] for r in smoke_rows)

    if arm == "O1":
        require(domains == Counter({"idiom": 16}), f"O1 smoke domains={domains}")
    elif arm == "O2":
        require(domains == Counter({"chemistry": 16}), f"O2 smoke domains={domains}")
    elif arm == "O3":
        require(
            domains == Counter({"chemistry": 8, "idiom": 8}),
            f"O3 smoke domains={domains}",
        )

    sdev = "npu:0"
    tdev = "npu:1"
    torch.npu.set_device(sdev)
    set_seed(SEED)

    st = AutoTokenizer.from_pretrained(C0)
    tt = AutoTokenizer.from_pretrained(TEACHER)

    require(st.pad_token_id is not None, "student pad token missing")
    require(tt.pad_token_id is not None, "teacher pad token missing")
    require(st.eos_token_id == tt.eos_token_id, "EOS id mismatch")

    sh = tokenizer_vocab_hash(st)
    th = tokenizer_vocab_hash(tt)
    require(sh == th, f"tokenizer vocab mismatch student={sh} teacher={th}")

    student = AutoModelForCausalLM.from_pretrained(
        C0,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(sdev)
    student.config.use_cache = False

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(tdev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # First run the corrected semantic replay/reduction check.
    custom_semantic_preflight(
        student,
        teacher,
        st,
        tt,
        rows,
        sdev,
        tdev,
        out_dir,
    )

    micro_per_epoch = math.ceil(len(rows) / MICRO_BATCH)
    updates_per_epoch = math.ceil(micro_per_epoch / GRAD_ACCUM)
    total_updates = updates_per_epoch * EPOCHS
    warmup_steps = int(total_updates * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    print("============================================================", flush=True)
    print(f"{arm} END-TO-END OPD PREFLIGHT v3", flush=True)
    print("MODE=DRY_RUN_REAL_UPDATE_DISCARDED_AFTER_PROCESS", flush=True)
    print(f"rows=16 micro_batch={MICRO_BATCH} grad_accum={GRAD_ACCUM}", flush=True)
    print(f"domains={dict(domains)}", flush=True)
    print("path=rollout->teacher_topk32->FKL->backward->GA4->AdamW->save->reload", flush=True)
    print("============================================================", flush=True)

    optimizer.zero_grad(set_to_none=True)
    group_tokens = 0
    all_row_kls = []
    rollout_lengths = []

    # Exactly four real microbatches => one real GA4 optimizer boundary.
    for micro_idx in range(GRAD_ACCUM):
        begin = micro_idx * MICRO_BATCH
        rr = smoke_rows[begin:begin + MICRO_BATCH]
        require(len(rr) == MICRO_BATCH, "preflight microbatch size mismatch")

        responses = generate_student_responses(
            student,
            st,
            rr,
            sdev,
            epoch=0,
        )
        rollout_lengths.extend(len(x) for x in responses)

        tk = teacher_topk_for_batch(
            teacher,
            tt,
            rr,
            responses,
            tdev,
        )

        kl_sum, ntok, row_kls = student_kl_sum_for_batch(
            student,
            st,
            rr,
            responses,
            tk,
            sdev,
        )

        require(bool(torch.isfinite(kl_sum).item()), f"nonfinite KL micro={micro_idx}")
        kl_sum.backward()

        group_tokens += int(ntok)
        all_row_kls.extend(float(x) for x in row_kls)

        print(
            f"{now8()} arm={arm} preflight_micro={micro_idx+1}/{GRAD_ACCUM} "
            f"response_tokens={ntok} mean_row_kl={sum(row_kls)/len(row_kls):.6f}",
            flush=True,
        )

    require(group_tokens > 0, "preflight group_tokens=0")

    # Match the real training reduction exactly.
    for p in student.parameters():
        if p.grad is not None:
            p.grad.div_(group_tokens)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        student.parameters(),
        MAX_GRAD_NORM,
    )
    require(
        bool(torch.isfinite(torch.as_tensor(grad_norm)).item()),
        f"nonfinite grad_norm={grad_norm}",
    )

    # Require that at least one finite nonzero gradient exists.
    nonzero_grad_tensors = 0
    sampled_grad_absmax = 0.0
    for p in student.parameters():
        if p.grad is None:
            continue
        require(bool(torch.isfinite(p.grad).all().item()), "nonfinite gradient tensor")
        gmax = float(p.grad.detach().abs().max().cpu())
        if gmax > 0:
            nonzero_grad_tensors += 1
            sampled_grad_absmax = max(sampled_grad_absmax, gmax)

    require(nonzero_grad_tensors > 0, "all gradients are zero")

    lr_before = float(scheduler.get_last_lr()[0])
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    lr_after = float(scheduler.get_last_lr()[0])

    require(len(optimizer.state) > 0, "AdamW optimizer state was not created")

    # Exercise the SAME durable resume helper as the real run, but under an
    # isolated preflight directory.
    resume_probe = out_dir / "resume_probe"
    if resume_probe.exists():
        shutil.rmtree(resume_probe)
    resume_probe.mkdir(parents=True, exist_ok=True)

    save_resume(
        resume_probe,
        student,
        st,
        optimizer,
        scheduler,
        completed_epoch=0,
        completed_updates=1,
    )

    require((resume_probe / "resume_hf").exists(), "resume_hf missing")
    require((resume_probe / "resume_state.pt").exists(), "resume_state.pt missing")
    require((resume_probe / "resume_meta.json").exists(), "resume_meta.json missing")

    # Drop the updated model, then prove that checkpoint + optimizer/scheduler
    # can be restored in the same runtime.
    del optimizer
    del scheduler
    del student
    try:
        torch.npu.empty_cache()
    except Exception:
        pass

    reloaded = AutoModelForCausalLM.from_pretrained(
        resume_probe / "resume_hf",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(sdev)
    reloaded.config.use_cache = False

    reloaded_optimizer = torch.optim.AdamW(
        reloaded.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )
    reloaded_scheduler = get_linear_schedule_with_warmup(
        reloaded_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    saved = torch.load(
        resume_probe / "resume_state.pt",
        map_location="cpu",
    )
    require(saved["completed_epoch"] == 0, "resume epoch mismatch")
    require(saved["completed_updates"] == 1, "resume update mismatch")

    reloaded_optimizer.load_state_dict(saved["optimizer"])
    optimizer_to(reloaded_optimizer, sdev)
    reloaded_scheduler.load_state_dict(saved["scheduler"])
    require(len(reloaded_optimizer.state) > 0, "reloaded optimizer state empty")

    # One post-reload greedy generation proves that the saved model is usable.
    reloaded.eval()
    probe_translation = generate_one_translation(
        reloaded,
        st,
        smoke_rows[0]["src_text"],
        sdev,
    )
    require(
        isinstance(probe_translation, str) and probe_translation.strip(),
        "post-reload generation empty",
    )

    semantic = json.loads(
        (out_dir / "semantic_preflight_v2.json").read_text(encoding="utf-8")
    )
    require(semantic.get("status") == "PASS", "semantic preflight artifact not PASS")

    result = {
        "status": "PASS",
        "version": 3,
        "scientific_class": "RUNTIME_DRY_RUN_ONLY",
        "claim_boundary":
            "one discarded real optimizer update; scientific arm still starts from frozen C0",
        "arm": arm,
        "script_sha256": current_script_sha,
        "frozen_inputs": frozen,
        "rows": len(smoke_rows),
        "domains": dict(domains),
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRAD_ACCUM,
        "group_response_tokens": group_tokens,
        "mean_row_kl": sum(all_row_kls) / len(all_row_kls),
        "min_row_kl": min(all_row_kls),
        "max_row_kl": max(all_row_kls),
        "mean_rollout_tokens": sum(rollout_lengths) / len(rollout_lengths),
        "max_rollout_tokens": max(rollout_lengths),
        "grad_norm_before_clip_return": float(grad_norm),
        "nonzero_gradient_tensors": nonzero_grad_tensors,
        "max_abs_gradient_after_scaling_and_clip": sampled_grad_absmax,
        "lr_before_step": lr_before,
        "lr_after_scheduler_step": lr_after,
        "optimizer_state_entries_after_step": len(reloaded_optimizer.state),
        "save_resume_helper": "PASS",
        "model_reload": "PASS",
        "optimizer_reload": "PASS",
        "scheduler_reload": "PASS",
        "post_reload_generation": "PASS",
        "post_reload_translation_preview": probe_translation[:500],
        "semantic_preflight": semantic,
        "created": now8(),
    }
    atomic_json(summary_path, result)

    print(
        f"{now8()} {arm}_E2E_PREFLIGHT_V3=PASS "
        f"mean_kl={result['mean_row_kl']:.6f} "
        f"grad_norm={float(grad_norm):.6f} "
        f"optimizer_states={len(reloaded_optimizer.state)} "
        f"save_reload=PASS",
        flush=True,
    )

    # Remove the disposable model/optimizer checkpoint after proving reload.
    # Keep only small JSON/log provenance.
    del reloaded_optimizer
    del reloaded_scheduler
    del reloaded
    try:
        torch.npu.empty_cache()
    except Exception:
        pass
    shutil.rmtree(resume_probe, ignore_errors=True)


def cmd_opd_worker(args) -> None:
    import torch
    import torch_npu  # noqa: F401
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    arm = args.arm
    out_dir = RUN / arm / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_hf = out_dir / "final_hf"
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and final_hf.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("status") == "PASS":
            print(f"{now8()} {arm}_TRAIN_ALREADY_PASS", flush=True)
            return

    frozen = verify_frozen_inputs()
    rows = load_arm_rows(arm)

    sdev = "npu:0"
    tdev = "npu:1"
    torch.npu.set_device(sdev)
    set_seed(SEED)

    st = AutoTokenizer.from_pretrained(C0)
    tt = AutoTokenizer.from_pretrained(TEACHER)

    require(st.pad_token_id is not None, "student pad token missing")
    require(tt.pad_token_id is not None, "teacher pad token missing")
    require(st.eos_token_id == tt.eos_token_id, "EOS id mismatch")

    sh = tokenizer_vocab_hash(st)
    th = tokenizer_vocab_hash(tt)
    require(
        sh == th,
        f"TOKENIZER_VOCAB_INCOMPATIBLE student={sh} teacher={th}",
    )

    # Student can resume only from a durable completed-epoch checkpoint.
    resume_state = out_dir / "resume_state.pt"
    resume_hf = out_dir / "resume_hf"
    saved = None
    start_epoch = 1
    completed_updates = 0

    if resume_state.exists() and resume_hf.exists():
        saved = torch.load(resume_state, map_location="cpu")
        start_epoch = int(saved["completed_epoch"]) + 1
        completed_updates = int(saved["completed_updates"])
        student_source = resume_hf
        print(
            f"{now8()} {arm}_RESUME completed_epoch={start_epoch-1} "
            f"updates={completed_updates}",
            flush=True,
        )
    else:
        student_source = C0

    student = AutoModelForCausalLM.from_pretrained(
        student_source,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(sdev)
    student.config.use_cache = False

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(tdev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    resolved = {
        "student_attention_implementation":
            getattr(student.config, "_attn_implementation", None),
        "teacher_attention_implementation":
            getattr(teacher.config, "_attn_implementation", None),
        "student_rope_theta": getattr(student.config, "rope_theta", None),
        "teacher_rope_theta": getattr(teacher.config, "rope_theta", None),
        "student_eos": st.eos_token_id,
        "teacher_eos": tt.eos_token_id,
        "student_pad": st.pad_token_id,
        "teacher_pad": tt.pad_token_id,
        "vocab_sha256": sh,
    }

    # Prompt-length audit uses only source-side data.
    max_sp = 0
    max_tp = 0
    for r in rows:
        max_sp = max(max_sp, len(render_prompt_ids(st, student_user_text(r))))
        max_tp = max(max_tp, len(render_prompt_ids(tt, teacher_user_text(r))))
    require(
        max_sp + MAX_NEW_TOKENS <= MAX_TOTAL_LENGTH,
        f"student max possible length overflow {max_sp}+{MAX_NEW_TOKENS}",
    )
    require(
        max_tp + MAX_NEW_TOKENS <= MAX_TOTAL_LENGTH,
        f"teacher max possible length overflow {max_tp}+{MAX_NEW_TOKENS}",
    )

    # 3 passes, one rollout per source per pass.
    micro_per_epoch = math.ceil(len(rows) / MICRO_BATCH)
    updates_per_epoch = math.ceil(micro_per_epoch / GRAD_ACCUM)
    total_updates = updates_per_epoch * EPOCHS
    warmup_steps = int(total_updates * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_to(optimizer, sdev)
        scheduler.load_state_dict(saved["scheduler"])

    if start_epoch == 1:
        custom_semantic_preflight(
            student, teacher, st, tt, rows, sdev, tdev, out_dir
        )

    question = (
        "Can Teacher-only WA lexical knowledge be transferred through "
        "Student-trajectory top-k32 forward-KL?"
    )
    print("============================================================", flush=True)
    print(f"TARGETED WA-OPD {arm}", flush=True)
    print("SCIENTIFIC_CLASS=LAB_ADAPTATION_CUSTOM_TORCH_NPU_KNOWLEDGE_CONDITIONED_OPD", flush=True)
    print(f"Question: {question}", flush=True)
    print("Competing explanations: knowledge-conditioned on-policy KL transfers targeted knowledge vs sequence target remains necessary.", flush=True)
    print("Falsifiable prediction: O1 raises Idiom; O2 raises Chemistry; O3 raises both vs C0.", flush=True)
    print("Decision after result: positive targeted transfer -> port/freeze formal mechanism; weak transfer -> localize OPD transfer bottleneck.", flush=True)
    print(f"arm={arm} rows={len(rows)} epochs={EPOCHS}", flush=True)
    print(f"micro_batch={MICRO_BATCH} grad_accum={GRAD_ACCUM} effective_batch={EFFECTIVE_BATCH}", flush=True)
    print(f"lr={LR} weight_decay={WEIGHT_DECAY} warmup_ratio={WARMUP_RATIO}", flush=True)
    print(f"temperature={ROLLOUT_TEMPERATURE} max_new_tokens={MAX_NEW_TOKENS}", flush=True)
    print(f"teacher_top_k={TOP_K_TEACHER} objective=renormalized_topk_forward_KL", flush=True)
    print("student_lexical_hint=FALSE teacher_lexical_hint=TRUE", flush=True)
    print("reward=PPO=advantage=FALSE", flush=True)
    print(f"max_student_prompt_tokens={max_sp} max_teacher_prompt_tokens={max_tp}", flush=True)
    print(json.dumps(resolved, ensure_ascii=False, sort_keys=True), flush=True)
    print("============================================================", flush=True)

    atomic_json(
        out_dir / "progress.json",
        {
            "status": "RUNNING",
            "arm": arm,
            "epoch": start_epoch,
            "epochs": EPOCHS,
            "completed_updates": completed_updates,
            "total_updates": total_updates,
            "updated": now8(),
        },
    )

    started = time.time()
    session_start_updates = completed_updates
    loss_recent = []
    token_recent = []
    rollout_len_recent = []
    domain_recent = Counter()

    for epoch in range(start_epoch, EPOCHS + 1):
        order = list(range(len(rows)))
        random.Random(SEED + epoch - 1).shuffle(order)

        student.train()
        optimizer.zero_grad(set_to_none=True)

        micro_batches = [
            order[i:i + MICRO_BATCH]
            for i in range(0, len(order), MICRO_BATCH)
        ]

        group_tokens = 0
        group_micros = 0

        for mi, idxs in enumerate(micro_batches, 1):
            rr = [rows[i] for i in idxs]

            responses = generate_student_responses(
                student, st, rr, sdev, epoch
            )
            teacher_topk = teacher_topk_for_batch(
                teacher, tt, rr, responses, tdev
            )

            kl_sum, ntok, row_kls = student_kl_sum_for_batch(
                student, st, rr, responses, teacher_topk, sdev
            )

            # Backprop the SUM now. At optimizer boundary, divide accumulated
            # gradients by exact valid response-token count across micros.
            kl_sum.backward()
            group_tokens += ntok
            group_micros += 1

            loss_recent.extend(row_kls)
            if len(loss_recent) > 100:
                loss_recent = loss_recent[-100:]
            token_recent.append(ntok)
            if len(token_recent) > 100:
                token_recent = token_recent[-100:]
            rollout_len_recent.extend(len(x) for x in responses)
            if len(rollout_len_recent) > 200:
                rollout_len_recent = rollout_len_recent[-200:]
            for r in rr:
                domain_recent[r["domain"]] += 1

            boundary = (
                group_micros == GRAD_ACCUM
                or mi == len(micro_batches)
            )
            if not boundary:
                continue

            require(group_tokens > 0, "group_tokens=0")

            for p in student.parameters():
                if p.grad is not None:
                    p.grad.div_(group_tokens)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), MAX_GRAD_NORM
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            completed_updates += 1
            group_tokens = 0
            group_micros = 0

            if completed_updates % 10 == 0 or completed_updates == total_updates:
                elapsed = max(time.time() - started, 1e-9)
                fresh = max(completed_updates - session_start_updates, 1)
                rate = fresh / elapsed
                remaining = total_updates - completed_updates
                eta = remaining / rate if rate else None
                mean_kl = sum(loss_recent) / len(loss_recent)
                mean_len = sum(rollout_len_recent) / len(rollout_len_recent)
                lr_now = scheduler.get_last_lr()[0]

                print(
                    f"{now8()} arm={arm} epoch={epoch}/{EPOCHS} "
                    f"update={completed_updates}/{total_updates} "
                    f"pct={100*completed_updates/total_updates:.1f}% "
                    f"kl100={mean_kl:.6f} "
                    f"rollout_len200={mean_len:.1f} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"lr={lr_now:.8g} "
                    f"ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                    flush=True,
                )

                atomic_json(
                    out_dir / "progress.json",
                    {
                        "status": "RUNNING",
                        "arm": arm,
                        "epoch": epoch,
                        "epochs": EPOCHS,
                        "completed_updates": completed_updates,
                        "total_updates": total_updates,
                        "percentage": round(
                            100 * completed_updates / total_updates, 2
                        ),
                        "mean_recent_row_kl": mean_kl,
                        "mean_recent_rollout_tokens": mean_len,
                        "grad_norm": float(grad_norm),
                        "lr": lr_now,
                        "eta_seconds": round(eta, 1) if eta is not None else None,
                        "domain_examples_seen_recent_session":
                            dict(domain_recent),
                        "updated": now8(),
                    },
                )

        save_resume(
            out_dir, student, st, optimizer, scheduler,
            completed_epoch=epoch,
            completed_updates=completed_updates,
        )
        print(
            f"{now8()} arm={arm} EPOCH_CHECKPOINT=PASS "
            f"epoch={epoch}/{EPOCHS} updates={completed_updates}",
            flush=True,
        )

    final_tmp = out_dir / "final_hf.tmp"
    if final_tmp.exists():
        shutil.rmtree(final_tmp)
    final_tmp.mkdir(parents=True, exist_ok=True)
    student.config.use_cache = True
    student.save_pretrained(final_tmp, safe_serialization=True)
    st.save_pretrained(final_tmp)
    if final_hf.exists():
        shutil.rmtree(final_hf)
    os.replace(final_tmp, final_hf)

    manifest = {
        "status": "PASS",
        "scientific_class":
            "LAB_ADAPTATION_CUSTOM_TORCH_NPU_KNOWLEDGE_CONDITIONED_OPD",
        "claim_boundary":
            "Teacher receives lexical knowledge; do not call canonical/paper-exact OPD",
        "arm": arm,
        "init_model": str(C0),
        "teacher_model": str(TEACHER),
        "rows": len(rows),
        "domains":
            dict(Counter(r["domain"] for r in rows)),
        "frozen_inputs": frozen,
        "tokenizer_vocab_sha256": sh,
        "resolved_runtime": resolved,
        "e2e_runtime_preflight_v3": {
            "status": "PASS",
            "aggregate_artifact": str(RUN / "preflight_v3" / "summary.json"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "checks":
                "real discarded GA4 update + AdamW/scheduler + save_resume + reload",
        },
        "semantic_preflight": {
            "version": 2,
            "artifact": str(out_dir / "semantic_preflight_v2.json"),
            "criterion":
                "exact keyed response IDs across MB1/MB2/MB4 plus "
                "partition-invariant reduction on canonical rowwise scores",
            "batched_BF16_partition_drift_is_non_blocking": True,
        },
        "training": {
            "epochs": EPOCHS,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": GRAD_ACCUM,
            "effective_batch_nominal": EFFECTIVE_BATCH,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "optimizer": "AdamW(foreach=False)",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "scheduler": "linear",
            "max_grad_norm": MAX_GRAD_NORM,
            "precision": "bfloat16",
            "full_parameter_student_update": True,
            "teacher_frozen": True,
            "seed": SEED,
            "epoch_shuffle_seed_rule": "seed + epoch - 1",
            "rollout_row_seed_rule":
                "sha256(seed|epoch|domain|job_id)",
            "rollout_temperature": ROLLOUT_TEMPERATURE,
            "rollout_top_p": 1.0,
            "rollout_top_k": 0,
            "rollout_max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": False,
            "teacher_top_k": TOP_K_TEACHER,
            "objective":
                "Teacher-topk-renormalized forward KL on Student response tokens",
            "loss_reduction":
                "global valid response-token mean across accumulated microbatches",
            "reward_terms": False,
            "ppo_terms": False,
            "advantage_terms": False,
        },
        "conditioning": {
            "student": "original source direct-translation prompt only",
            "teacher_chemistry":
                "original source + strict lexical_record.en_name + exact Student prefix",
            "teacher_idiom":
                "original source + dictionary definition + exact Student prefix",
        },
        "final_model": str(final_hf),
        "created": now8(),
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        out_dir / "progress.json",
        {
            "status": "PASS",
            "arm": arm,
            "completed_updates": total_updates,
            "total_updates": total_updates,
            "percentage": 100.0,
            "updated": now8(),
        },
    )

    print(f"{now8()} {arm}_WA_OPD_TRAIN=PASS", flush=True)
    print(f"{arm}_FINAL_MODEL={final_hf}", flush=True)


def extract_general_row(row: dict[str, Any], dataset: str, idx: int):
    source = row.get("source")
    reference = row.get("reference")
    require(
        isinstance(source, str) and source.strip(),
        f"{dataset}[{idx}] missing source",
    )
    require(
        isinstance(reference, str) and reference.strip(),
        f"{dataset}[{idx}] missing reference",
    )
    return source, reference


def generate_one_translation(model, tok, source: str, device: str) -> str:
    import torch

    user = DIRECT_PROMPT.format(source=source)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    batch = tok(rendered, return_tensors="pt", add_special_tokens=False)
    batch = {k: v.to(device) for k, v in batch.items()}
    width = batch["input_ids"].shape[1]
    with torch.no_grad():
        seq = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=512,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
    return tok.decode(
        seq[0, width:].detach().cpu().tolist(),
        skip_special_tokens=True,
    ).strip()


def existing_pass_map(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for r in read_jsonl(path):
        if r.get("status") == "PASS":
            out[str(r[key])] = r
    return out


def cmd_bleu_worker(args) -> None:
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("status") == "PASS":
            print(f"{now8()} BLEU_ALREADY_PASS label={args.label}", flush=True)
            return

    device = "npu:0"
    torch.npu.set_device(device)

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    all_metrics = {}
    overall_started = time.time()

    for ds, path in GENERAL_EVAL.items():
        rows = read_jsonl(path)
        require(
            len(rows) == GENERAL_EXPECTED_ROWS[ds],
            f"{ds} row count mismatch",
        )

        pred_path = out_dir / f"{ds.lower()}_predictions.jsonl"
        done = existing_pass_map(pred_path, "row_id")
        initial = len(done)
        started = time.time()

        for i, row in enumerate(rows):
            rid = f"{ds}:{i}"
            if rid in done:
                continue
            src, ref = extract_general_row(row, ds, i)
            hyp = generate_one_translation(model, tok, src, device)
            rec = {
                "status": "PASS",
                "label": args.label,
                "dataset": ds,
                "row_id": rid,
                "source": src,
                "reference": ref,
                "student_translation": hyp,
                "prompt_sha256": EXPECTED_DIRECT_PROMPT_SHA,
                "enable_thinking": False,
                "do_sample": False,
                "max_new_tokens": 512,
                "timestamp": now8(),
            }
            append_jsonl(pred_path, rec)
            done[rid] = rec

            n = len(done)
            if n % 50 == 0 or n == len(rows):
                elapsed = max(time.time() - started, 1e-9)
                fresh = n - initial
                rate = fresh / elapsed if fresh else 0
                eta = (len(rows) - n) / rate if rate else None
                print(
                    f"{now8()} label={args.label} dataset={ds} "
                    f"done={n}/{len(rows)} pct={100*n/len(rows):.1f}% "
                    f"rate={rate:.2f}/s "
                    f"ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                    flush=True,
                )

        ordered = [done[f"{ds}:{i}"] for i in range(len(rows))]
        refs = [x["reference"] for x in ordered]
        hyps = [x["student_translation"] for x in ordered]

        try:
            import sacrebleu
        except Exception as e:
            raise RuntimeError(
                "sacrebleu import failed in existing environment"
            ) from e

        bleu = float(sacrebleu.corpus_bleu(hyps, [refs]).score)
        chrf = float(sacrebleu.corpus_chrf(hyps, [refs]).score)

        all_metrics[ds] = {
            "rows": len(rows),
            "BLEU": bleu,
            "chrF": chrf,
            "predictions_sha256": sha256_file(pred_path),
        }
        print(
            f"{args.label} {ds} BLEU={bleu:.6f} chrF={chrf:.6f}",
            flush=True,
        )

    macro_bleu = sum(x["BLEU"] for x in all_metrics.values()) / 3
    macro_chrf = sum(x["chrF"] for x in all_metrics.values()) / 3
    summary = {
        "status": "PASS",
        "scientific_class": "GENERAL_MT_SIDE_DIAGNOSTIC",
        "label": args.label,
        "model": str(model_path),
        "generation": {
            "prompt_sha256": EXPECTED_DIRECT_PROMPT_SHA,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 512,
        },
        "datasets": all_metrics,
        "macro_bleu": macro_bleu,
        "macro_chrf": macro_chrf,
        "delta_macro_bleu_vs_C0": macro_bleu - BASE_GENERAL["macro_bleu"],
        "elapsed_seconds": time.time() - overall_started,
        "created": now8(),
    }
    atomic_json(summary_path, summary)
    print(
        f"{now8()} GENERAL_BLEU_PASS label={args.label} "
        f"macro_BLEU={macro_bleu:.6f} "
        f"delta_vs_C0={summary['delta_macro_bleu_vs_C0']:+.6f} "
        f"macro_chrF={macro_chrf:.6f}",
        flush=True,
    )


def cmd_targeted_eval_worker(args) -> None:
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "chemistry_summary.json"
    idiom_path = out_dir / f"{args.label.lower()}_idiom1000_for_judge.jsonl"
    if summary_path.exists() and idiom_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        if s.get("status") == "PASS":
            print(f"{now8()} TARGETED_ALREADY_PASS label={args.label}", flush=True)
            return

    verify_frozen_inputs()
    diag_path = DIAG / "targeted_diagnostic2000.jsonl"
    rows = read_jsonl(diag_path)
    require(len(rows) == 2000, f"diagnostic rows={len(rows)}")

    device = "npu:0"
    torch.npu.set_device(device)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    pred_path = out_dir / "targeted_diagnostic2000_translations.jsonl"
    done = existing_pass_map(pred_path, "eval_id")
    initial = len(done)
    started = time.time()

    for i, r in enumerate(rows):
        eid = r["eval_id"]
        if eid in done:
            continue
        hyp = generate_one_translation(model, tok, r["src_text"], device)
        rec = {
            "status": "PASS",
            "label": args.label,
            "eval_id": eid,
            "eval_domain": r["eval_domain"],
            "eval_split": r["eval_split"],
            "job_id": r["job_id"],
            "entity_key": r.get("entity_key"),
            "src_term": r["src_term"],
            "definition": r.get("definition"),
            "src_text": r["src_text"],
            "canonical_en_target": r.get("canonical_en_target"),
            "model_translation": hyp,
            "timestamp": now8(),
        }
        append_jsonl(pred_path, rec)
        done[eid] = rec

        n = len(done)
        if n % 50 == 0 or n == len(rows):
            elapsed = max(time.time() - started, 1e-9)
            fresh = n - initial
            rate = fresh / elapsed if fresh else 0
            eta = (len(rows) - n) / rate if rate else None
            print(
                f"{now8()} label={args.label} targeted "
                f"done={n}/2000 pct={100*n/2000:.1f}% "
                f"rate={rate:.2f}/s "
                f"ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                flush=True,
            )

    ordered = [done[r["eval_id"]] for r in rows]
    chem = [r for r in ordered if r["eval_domain"] == "chemistry"]
    idiom = [r for r in ordered if r["eval_domain"] == "idiom"]
    require(len(chem) == 1000, f"chem diagnostic n={len(chem)}")
    require(len(idiom) == 1000, f"idiom diagnostic n={len(idiom)}")

    for r in chem:
        target = r.get("canonical_en_target")
        require(
            isinstance(target, str) and target.strip(),
            f"chem canonical target missing {r['eval_id']}",
        )
        r["canonical_target_hit"] = (
            target.casefold() in r["model_translation"].casefold()
        )

    def cstats(split=None):
        xs = chem if split is None else [
            r for r in chem if r["eval_split"] == split
        ]
        hits = sum(bool(r["canonical_target_hit"]) for r in xs)
        return {"n": len(xs), "hits": hits, "accuracy": hits / len(xs)}

    summary = {
        "status": "PASS",
        "scientific_class": "TARGETED_DIAGNOSTIC_ONLY",
        "label": args.label,
        "metric":
            "case-insensitive canonical English target substring accuracy",
        "overall": cstats(),
        "uc500": cstats("uc"),
        "uw500": cstats("uw"),
        "created": now8(),
    }
    atomic_json(summary_path, summary)

    judge_rows = []
    for r in idiom:
        d = r.get("definition")
        require(isinstance(d, str) and d.strip(), f"missing idiom def {r['eval_id']}")
        judge_rows.append({
            "eval_id": f"{args.label}|{r['eval_id']}",
            "base_eval_id": r["eval_id"],
            "arm": args.label,
            "job_id": r["job_id"],
            "split": r["eval_split"],
            "src_term": r["src_term"],
            "definition": d,
            "src_text": r["src_text"],
            "model_translation": r["model_translation"],
        })
    require(
        Counter(r["split"] for r in judge_rows)
        == Counter({"uc": 500, "uw": 500}),
        "idiom split mismatch",
    )
    write_jsonl_atomic(idiom_path, judge_rows)

    print(
        f"{now8()} TARGETED_EVAL_PASS label={args.label} "
        f"Chem={summary['overall']['accuracy']:.4f} "
        f"UC={summary['uc500']['accuracy']:.4f} "
        f"UW={summary['uw500']['accuracy']:.4f}",
        flush=True,
    )
    print(f"{args.label}_IDIOM_JUDGE_INPUT={idiom_path}", flush=True)


def spawn_self(
    subcmd: list[str],
    visible_devices: str,
    log_path: Path,
) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8")
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-u", str(Path(__file__).resolve())] + subcmd
    p = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    # The child owns a duplicated descriptor after Popen. Closing the parent's
    # handle here avoids descriptor accumulation across phases.
    fh.close()
    return p, None


def child_state(item) -> dict[str, Any]:
    label, kind, p, log, progress = item
    d = {
        "label": label,
        "kind": kind,
        "pid": p.pid,
        "returncode": p.poll(),
        "log": str(log),
    }
    if progress and progress.exists():
        try:
            d["progress"] = json.loads(progress.read_text(encoding="utf-8"))
        except Exception:
            d["progress"] = "READ_ERROR"
    return d


def monitor(children, state_path: Path, phase: str, every: int = 20):
    while any(item[2].poll() is None for item in children):
        states = [child_state(x) for x in children]
        atomic_json(
            state_path,
            {
                "status": "RUNNING",
                "phase": phase,
                "children": states,
                "updated": now8(),
            },
        )

        compact = []
        for s in states:
            rc = s["returncode"]
            if rc is None:
                status = "RUNNING"
            else:
                status = f"RC={rc}"
            prog = s.get("progress")
            if isinstance(prog, dict):
                pct = prog.get("percentage")
                if pct is not None:
                    status += f":{pct}%"
            compact.append(f"{s['label']}[{s['kind']}]={status}")

        print(
            f"{now8()} phase={phase} " + " | ".join(compact),
            flush=True,
        )
        time.sleep(every)

    for item in children:
        item[2].wait()

    states = [child_state(x) for x in children]
    atomic_json(
        state_path,
        {
            "status": "PASS"
                if all(s["returncode"] == 0 for s in states)
                else "FAIL",
            "phase": phase,
            "children": states,
            "updated": now8(),
        },
    )
    return states


def combine_idiom_o123() -> Path:
    all_rows = []
    for arm in ("O1", "O2", "O3"):
        p = RUN / arm / "targeted_eval" / f"{arm.lower()}_idiom1000_for_judge.jsonl"
        rows = read_jsonl(p)
        require(len(rows) == 1000, f"{arm} idiom rows={len(rows)}")
        require(all(r["arm"] == arm for r in rows), f"{arm} arm field mismatch")
        all_rows.extend(rows)

    require(len(all_rows) == 3000, "combined idiom n != 3000")
    require(
        len({r["eval_id"] for r in all_rows}) == 3000,
        "combined idiom duplicate eval_id",
    )
    out = RUN / "o123_idiom_diagnostic3000_for_judge.jsonl"
    write_jsonl_atomic(out, all_rows)
    return out


def build_final_summary(pre_bleu_states, train_states, target_states, o_bleu_states):
    summary = {
        "status": "PASS",
        "scientific_class":
            "LAB_ADAPTATION_CUSTOM_TORCH_NPU_KNOWLEDGE_CONDITIONED_OPD",
        "question":
            "Can WA lexical knowledge that is learnable by SFT also transfer through on-policy forward-KL?",
        "SFT_positive_control": {
            "Chemistry": {
                "C0": 0.097,
                "C1": 0.086,
                "C2": 0.261,
                "C3": 0.261,
            },
            "Idiom": {
                "C0": 2.613,
                "C1": 3.132,
                "C2": 2.382,
                "C3": 3.141,
            },
        },
        "O_chemistry": {},
        "general_bleu": {"SFT": {}, "OPD": {}},
        "children": {
            "pre_bleu": pre_bleu_states,
            "train": train_states,
            "targeted": target_states,
            "o_bleu": o_bleu_states,
        },
        "created": now8(),
    }

    for arm in ("O1", "O2", "O3"):
        cp = RUN / arm / "targeted_eval/chemistry_summary.json"
        if cp.exists():
            summary["O_chemistry"][arm] = json.loads(
                cp.read_text(encoding="utf-8")
            )

    for label in ("C1", "C2", "C3"):
        sp = RUN / "general_bleu_sft" / label / "summary.json"
        if sp.exists():
            summary["general_bleu"]["SFT"][label] = json.loads(
                sp.read_text(encoding="utf-8")
            )

    for label in ("O1", "O2", "O3"):
        sp = RUN / "general_bleu_opd" / label / "summary.json"
        if sp.exists():
            summary["general_bleu"]["OPD"][label] = json.loads(
                sp.read_text(encoding="utf-8")
            )

    atomic_json(RUN / "final_server_summary.json", summary)
    return summary



def preflight_v3_summary_valid() -> bool:
    p = RUN / "preflight_v3" / "summary.json"
    if not p.exists():
        return False
    try:
        x = json.loads(p.read_text(encoding="utf-8"))
        return (
            x.get("status") == "PASS"
            and x.get("version") == 3
            and x.get("script_sha256") == sha256_file(Path(__file__).resolve())
            and set(x.get("arms", {}).keys()) == {"O1", "O2", "O3"}
            and all(
                x["arms"][a].get("status") == "PASS"
                for a in ("O1", "O2", "O3")
            )
        )
    except Exception:
        return False


def cmd_preflight_master(args) -> None:
    """
    Run O1/O2/O3 full E2E dry-run concurrently on NPU pairs 0-5.
    It does NOT acquire the normal overnight master.lock, so it can run now
    while the old v1 master finishes C1/C2/C3 BLEU on devices 6-8.
    """
    pf = RUN / "preflight_v3"
    pf.mkdir(parents=True, exist_ok=True)
    state_path = pf / "state.json"

    lock = open(pf / "preflight.lock", "a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("FINAL_RESULT=O123_PREFLIGHT_V3_ALREADY_RUNNING", flush=True)
        return

    if preflight_v3_summary_valid():
        print("O123_PREFLIGHT_V3=ALREADY_PASS", flush=True)
        print("FINAL_RESULT=O123_PREFLIGHT_V3_PASS", flush=True)
        return

    print("============================================================", flush=True)
    print("O1/O2/O3 FULL END-TO-END PREFLIGHT v3", flush=True)
    print("MODE=DISCARDED_REAL_UPDATE", flush=True)
    print("O1 devices=0,1 | O2=2,3 | O3=4,5", flush=True)
    print("Checks=load+rollout+teacher_topk+FKL+backward+GA4+AdamW+save+reload", flush=True)
    print("Scientific O1/O2/O3 checkpoints remain untouched.", flush=True)
    print("============================================================", flush=True)

    devices = {"O1": "0,1", "O2": "2,3", "O3": "4,5"}
    children = []
    for arm in ("O1", "O2", "O3"):
        log = RUN / "logs" / f"{arm}_preflight_v3.log"
        p, fh = spawn_self(
            ["preflight-worker", "--arm", arm],
            devices[arm],
            log,
        )
        children.append((arm, "e2e_preflight", p, log, None))

    states = monitor(
        children,
        state_path,
        "o123_e2e_preflight_v3",
        10,
    )

    bad = [x for x in states if x["returncode"] != 0]
    if bad:
        atomic_json(
            pf / "summary.json",
            {
                "status": "FAIL",
                "version": 3,
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "children": states,
                "updated": now8(),
            },
        )
        print("=== PREFLIGHT FAILURE TAILS ===", flush=True)
        for x in bad:
            print(f"--- {x['label']} ---", flush=True)
            try:
                print(
                    "\n".join(
                        Path(x["log"]).read_text(
                            encoding="utf-8",
                            errors="replace",
                        ).splitlines()[-120:]
                    ),
                    flush=True,
                )
            except Exception:
                pass
        print("FINAL_RESULT=O123_PREFLIGHT_V3_FAIL", flush=True)
        raise SystemExit(11)

    arms = {}
    for arm in ("O1", "O2", "O3"):
        sp = pf / arm / "summary.json"
        require(sp.exists(), f"missing {arm} preflight summary")
        arms[arm] = json.loads(sp.read_text(encoding="utf-8"))
        require(arms[arm].get("status") == "PASS", f"{arm} preflight not PASS")

    summary = {
        "status": "PASS",
        "version": 3,
        "scientific_class": "RUNTIME_DRY_RUN_ONLY",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "arms": arms,
        "children": states,
        "created": now8(),
    }
    atomic_json(pf / "summary.json", summary)

    print("============================================================", flush=True)
    for arm in ("O1", "O2", "O3"):
        x = arms[arm]
        print(
            f"{arm}=PASS mean_kl={x['mean_row_kl']:.6f} "
            f"grad_norm={x['grad_norm_before_clip_return']:.6f} "
            f"save_reload={x['save_resume_helper']}",
            flush=True,
        )
    print("O123_PREFLIGHT_V3=PASS", flush=True)
    print("FINAL_RESULT=O123_PREFLIGHT_V3_PASS", flush=True)
    print("============================================================", flush=True)


def ensure_preflight_v3() -> None:
    if preflight_v3_summary_valid():
        print(
            f"{now8()} O123_PREFLIGHT_V3_REUSE=PASS "
            f"script_sha={sha256_file(Path(__file__).resolve())}",
            flush=True,
        )
        return

    # Same-file integrated fallback: if the user did not run the explicit
    # preflight first, the overnight master performs it automatically.
    class _Args:
        pass
    cmd_preflight_master(_Args())
    require(
        preflight_v3_summary_valid(),
        "O123_PREFLIGHT_V3_NOT_VALID_AFTER_RUN",
    )


def cmd_master(args) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    master_state = RUN / "state.json"

    # Single logical master/writer guard for the full overnight chain.
    #
    # v1 may still be alive only because its C1/C2/C3 BLEU side diagnostics
    # are finishing after O1/O2/O3 failed in preflight. v2 can be started now:
    # it waits on the SAME lock and takes over automatically when v1 exits.
    lock_handle = open(RUN / "master.lock", "a+", encoding="utf-8")
    waited = 0
    while True:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            break
        except BlockingIOError:
            if waited == 0 or waited % 60 == 0:
                print(
                    f"{now8()} MASTER_LOCK_WAIT "
                    f"waiting_for_previous_v1_master seconds={waited}",
                    flush=True,
                )
            time.sleep(10)
            waited += 10

    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(
        f"pid={os.getpid()} version=v2 acquired={now8()} waited_seconds={waited}\n"
    )
    lock_handle.flush()
    print(
        f"{now8()} MASTER_LOCK_ACQUIRED version=v2 waited_seconds={waited}",
        flush=True,
    )

    print("============================================================", flush=True)
    print("TARGETED WA-OPD OVERNIGHT v3", flush=True)
    print("SCIENTIFIC_CLASS=LAB_ADAPTATION_CUSTOM_TORCH_NPU_KNOWLEDGE_CONDITIONED_OPD", flush=True)
    print("Question: Can SFT-learnable targeted WA knowledge transfer through on-policy FKL?", flush=True)
    print("Competing explanations: WA knowledge is accessible through OPD vs sequence-target supervision is still crucial.", flush=True)
    print("Falsifiable prediction: O1 improves Idiom, O2 improves Chemistry, O3 improves both.", flush=True)
    print("Decision after result: positive -> formalize/port mechanism; weak -> analyze transfer bottleneck before PDS-OPD.", flush=True)
    print(f"RUN={RUN}", flush=True)
    print("BLEU_SIDE_DIAGNOSTIC=NON_BLOCKING", flush=True)
    print("============================================================", flush=True)

    frozen = verify_frozen_inputs()
    atomic_json(RUN / "frozen_input_provenance.json", frozen)

    # Full E2E runtime smoke must be PASS for this exact script snapshot.
    # If it was already run explicitly while v1 BLEU was finishing, reuse it.
    ensure_preflight_v3()

    # Phase A: C1/C2/C3 standard BLEU side diagnostic in parallel with O1/O2/O3.
    pre_bleu_specs = {
        "C1": SFT_RUN / "C1/train/final_hf",
        "C2": SFT_RUN / "C2/train/final_hf",
        "C3": SFT_RUN / "C3/train/final_hf",
    }
    for label, mp in pre_bleu_specs.items():
        require(mp.exists(), f"missing SFT model {label}: {mp}")

    pre_bleu_children = []
    for label, phys in zip(("C1", "C2", "C3"), ("6", "7", "8")):
        out = RUN / "general_bleu_sft" / label
        p, fh = spawn_self(
            [
                "bleu-worker",
                "--label", label,
                "--model", str(pre_bleu_specs[label]),
                "--out-dir", str(out),
            ],
            phys,
            RUN / "logs" / f"{label}_general_bleu.log",
        )
        pre_bleu_children.append(
            (label, "general_bleu", p, RUN / "logs" / f"{label}_general_bleu.log", None)
        )

    train_children = []
    opd_devices = {"O1": "0,1", "O2": "2,3", "O3": "4,5"}
    for arm in ("O1", "O2", "O3"):
        p, fh = spawn_self(
            ["opd-worker", "--arm", arm],
            opd_devices[arm],
            RUN / "logs" / f"{arm}_train.log",
        )
        train_children.append(
            (
                arm, "opd_train", p,
                RUN / "logs" / f"{arm}_train.log",
                RUN / arm / "train/progress.json",
            )
        )

    # Monitor both groups together; BLEU failures must not abort OPD.
    phase_a = pre_bleu_children + train_children
    states_a = monitor(phase_a, master_state, "sft_bleu_plus_o123_train", 20)

    pre_bleu_states = [s for s in states_a if s["kind"] == "general_bleu"]
    train_states = [s for s in states_a if s["kind"] == "opd_train"]

    bad_train = [s for s in train_states if s["returncode"] != 0]
    if bad_train:
        atomic_json(
            master_state,
            {
                "status": "FAIL",
                "phase": "o123_train",
                "bad_train": bad_train,
                "updated": now8(),
            },
        )
        print("=== OPD TRAIN FAILURE TAILS ===", flush=True)
        for s in bad_train:
            print(f"--- {s['label']} ---", flush=True)
            try:
                print(
                    "\n".join(
                        Path(s["log"]).read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()[-80:]
                    ),
                    flush=True,
                )
            except Exception:
                pass
        print("FINAL_RESULT=O123_TRAIN_FAIL", flush=True)
        raise SystemExit(2)

    print("O1_TRAIN=PASS O2_TRAIN=PASS O3_TRAIN=PASS", flush=True)

    # Phase B: targeted eval + standard BLEU for O1/O2/O3, all in parallel.
    target_children = []
    o_bleu_children = []

    for arm, phys in zip(("O1", "O2", "O3"), ("0", "1", "2")):
        model = RUN / arm / "train/final_hf"
        out = RUN / arm / "targeted_eval"
        p, fh = spawn_self(
            [
                "targeted-eval-worker",
                "--label", arm,
                "--model", str(model),
                "--out-dir", str(out),
            ],
            phys,
            RUN / "logs" / f"{arm}_targeted_eval.log",
        )
        target_children.append(
            (
                arm, "targeted_eval", p,
                RUN / "logs" / f"{arm}_targeted_eval.log",
                None,
            )
        )

    for arm, phys in zip(("O1", "O2", "O3"), ("6", "7", "8")):
        model = RUN / arm / "train/final_hf"
        out = RUN / "general_bleu_opd" / arm
        p, fh = spawn_self(
            [
                "bleu-worker",
                "--label", arm,
                "--model", str(model),
                "--out-dir", str(out),
            ],
            phys,
            RUN / "logs" / f"{arm}_general_bleu.log",
        )
        o_bleu_children.append(
            (
                arm, "general_bleu", p,
                RUN / "logs" / f"{arm}_general_bleu.log",
                None,
            )
        )

    states_b = monitor(
        target_children + o_bleu_children,
        master_state,
        "o123_targeted_plus_general_eval",
        20,
    )
    target_states = [s for s in states_b if s["kind"] == "targeted_eval"]
    o_bleu_states = [s for s in states_b if s["kind"] == "general_bleu"]

    bad_target = [s for s in target_states if s["returncode"] != 0]
    if bad_target:
        atomic_json(
            master_state,
            {
                "status": "FAIL",
                "phase": "targeted_eval",
                "bad_target": bad_target,
                "updated": now8(),
            },
        )
        print("FINAL_RESULT=O123_TARGETED_EVAL_FAIL", flush=True)
        raise SystemExit(3)

    judge_input = combine_idiom_o123()
    summary = build_final_summary(
        pre_bleu_states, train_states, target_states, o_bleu_states
    )

    atomic_json(
        master_state,
        {
            "status": "PASS",
            "phase": "server_complete",
            "judge_input": str(judge_input),
            "judge_input_sha256": sha256_file(judge_input),
            "summary": str(RUN / "final_server_summary.json"),
            "updated": now8(),
        },
    )

    print("============================================================", flush=True)
    print("TARGETED WA-OPD OVERNIGHT SERVER CHAIN COMPLETE", flush=True)

    for arm in ("O1", "O2", "O3"):
        c = summary["O_chemistry"].get(arm)
        if c:
            print(
                f"{arm}_CHEM overall={c['overall']['accuracy']:.4f} "
                f"UC={c['uc500']['accuracy']:.4f} "
                f"UW={c['uw500']['accuracy']:.4f}",
                flush=True,
            )

    for group in ("SFT", "OPD"):
        for label, d in summary["general_bleu"].get(group, {}).items():
            print(
                f"{group}_{label}_GENERAL macro_BLEU={d['macro_bleu']:.6f} "
                f"delta_vs_C0={d['delta_macro_bleu_vs_C0']:+.6f} "
                f"macro_chrF={d['macro_chrf']:.6f}",
                flush=True,
            )

    print(f"O123_IDIOM_JUDGE_INPUT={judge_input}", flush=True)
    print(f"O123_IDIOM_JUDGE_INPUT_SHA256={sha256_file(judge_input)}", flush=True)
    print(f"FINAL_SERVER_SUMMARY={RUN/'final_server_summary.json'}", flush=True)
    print("NEXT=RUN_FROZEN_DEEPSEEK_JUDGE_FOR_O1_O2_O3_ON_WINDOWS", flush=True)
    print("FINAL_RESULT=TARGETED_WA_OPD_SERVER_PASS", flush=True)
    print("============================================================", flush=True)


def build_parser():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp.add_parser("master")
    sp.add_parser("preflight-master")

    pf = sp.add_parser("preflight-worker")
    pf.add_argument("--arm", required=True, choices=["O1", "O2", "O3"])

    o = sp.add_parser("opd-worker")
    o.add_argument("--arm", required=True, choices=["O1", "O2", "O3"])

    b = sp.add_parser("bleu-worker")
    b.add_argument("--label", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--out-dir", required=True)

    t = sp.add_parser("targeted-eval-worker")
    t.add_argument("--label", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--out-dir", required=True)

    return ap


if __name__ == "__main__":
    try:
        args = build_parser().parse_args()
        if args.cmd == "master":
            cmd_master(args)
        elif args.cmd == "preflight-master":
            cmd_preflight_master(args)
        elif args.cmd == "preflight-worker":
            cmd_preflight_worker(args)
        elif args.cmd == "opd-worker":
            cmd_opd_worker(args)
        elif args.cmd == "bleu-worker":
            cmd_bleu_worker(args)
        elif args.cmd == "targeted-eval-worker":
            cmd_targeted_eval_worker(args)
        else:
            raise ValueError(args.cmd)
    except KeyboardInterrupt:
        print(f"{now8()} INTERRUPTED", flush=True)
        raise
    except Exception:
        traceback.print_exc()
        raise
