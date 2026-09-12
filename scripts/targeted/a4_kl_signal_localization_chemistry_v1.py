#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import time

from pathlib import Path


ROOT = Path("/workspace/mtpatcher")

REPO = ROOT / "repo/MT-Patcher-Reproduction-Ascend"

H5_RUN = (
    ROOT
    / "runs/targeted/wa_opd_horizon5_o12_20260911"
)

SOURCE = (
    REPO
    / "scripts/targeted/targeted_wa_opd_horizon5_o12_v2.py"
)

EXPECTED_SOURCE_SHA = (
    "38ac62c0a93f1b762b7536db60421dd8"
    "c1e7aca49a57c25851306c3cb76b6e16"
)

C0 = ROOT / "models/Qwen3-0.6B"
TEACHER = ROOT / "models/Qwen3-8B"

CONTEXT = (
    ROOT
    / "data/mtpatcher_v3_full6565_20260823"
    / "targeted_section43_contexts_qwen3_8b_final_20260910"
    / "chemistry_train.jsonl"
)

MANIFEST = (
    H5_RUN
    / "learning_curve/train_matched"
    / "chemistry_sample_manifest.json"
)

OUT = (
    ROOT
    / "runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)

    return h.hexdigest()


def require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def append_jsonl(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
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


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

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


def load_source():
    got = sha256_file(SOURCE)

    require(
        got == EXPECTED_SOURCE_SHA,
        f"source SHA mismatch got={got}",
    )

    spec = importlib.util.spec_from_file_location(
        "h5_source",
        SOURCE,
    )

    require(
        spec is not None and spec.loader is not None,
        "cannot load H5 source",
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def load_rows(n: int):
    all_rows = read_jsonl(CONTEXT)

    require(
        len(all_rows) == 5500,
        f"chemistry train rows={len(all_rows)}",
    )

    by_id = {
        int(r["job_id"]): r
        for r in all_rows
    }

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    ids = [
        int(x)
        for x in manifest["job_ids"]
    ]

    require(
        len(ids) == 1000,
        f"matched manifest n={len(ids)}",
    )

    ids = ids[:n]

    rows = [
        by_id[x]
        for x in ids
    ]

    for r in rows:
        require(
            r["domain"] == "chemistry",
            f"domain fail job={r['job_id']}",
        )

        en = (
            r["lexical_record"]
            ["en_name"]
            .strip()
        )

        require(
            bool(en),
            f"missing en_name job={r['job_id']}",
        )

    return rows


def existing_rows(path: Path):
    if not path.exists():
        return {}

    out = {}

    for r in read_jsonl(path):
        if r.get("status") == "PASS":
            out[int(r["job_id"])] = r

    return out


def teacher_nohint_logp_on_hint_support(
    teacher,
    teacher_tok,
    h5,
    rows,
    responses,
    hint_topk,
    device,
):
    import torch

    prompt_ids = [
        h5.render_prompt_ids(
            teacher_tok,
            h5.student_user_text(r),
        )
        for r in rows
    ]

    seqs = []

    for p, resp, row in zip(
        prompt_ids,
        responses,
        rows,
    ):
        require(
            len(p) + len(resp)
            <= h5.MAX_TOTAL_LENGTH,
            (
                "nohint teacher sequence too long "
                f"job={row['job_id']}"
            ),
        )

        seqs.append(
            p + resp
        )

    ids, mask = h5.pad_sequences(
        seqs,
        teacher_tok.pad_token_id,
        device,
    )

    with torch.no_grad():
        logits = teacher(
            input_ids=ids,
            attention_mask=mask,
        ).logits

    ret = []

    for i, (p, resp, tk) in enumerate(
        zip(
            prompt_ids,
            responses,
            hint_topk,
        )
    ):
        _, hint_indices_cpu = tk

        hint_indices = (
            hint_indices_cpu
            .to(device)
        )

        start = len(p) - 1
        end = start + len(resp)

        wanted = (
            logits[
                i,
                start:end,
                :
            ]
            .float()
        )

        lse = torch.logsumexp(
            wanted,
            dim=-1,
            keepdim=True,
        )

        selected = torch.gather(
            wanted,
            dim=-1,
            index=hint_indices,
        )

        logp = (
            selected - lse
        )

        ret.append(
            logp.cpu()
        )

    del logits
    del ids
    del mask

    return ret


def student_token_kl(
    student,
    student_tok,
    h5,
    rows,
    responses,
    hint_topk,
    device,
):
    import torch
    import torch.nn.functional as F

    prompt_ids = [
        h5.render_prompt_ids(
            student_tok,
            h5.student_user_text(r),
        )
        for r in rows
    ]

    seqs = [
        p + resp
        for p, resp in zip(
            prompt_ids,
            responses,
        )
    ]

    ids, mask = h5.pad_sequences(
        seqs,
        student_tok.pad_token_id,
        device,
    )

    with torch.no_grad():
        logits = student(
            input_ids=ids,
            attention_mask=mask,
        ).logits

    out = []

    for i, (p, resp, tk) in enumerate(
        zip(
            prompt_ids,
            responses,
            hint_topk,
        )
    ):
        tvals_cpu, tinds_cpu = tk

        tvals = tvals_cpu.to(device)
        tinds = tinds_cpu.to(device)

        start = len(p) - 1
        end = start + len(resp)

        slogits = (
            logits[
                i,
                start:end,
                :
            ]
            .float()
        )

        slse = torch.logsumexp(
            slogits,
            dim=-1,
            keepdim=True,
        )

        sselected = torch.gather(
            slogits,
            dim=-1,
            index=tinds,
        )

        slogp = (
            sselected - slse
        )

        tlogp = F.log_softmax(
            tvals.float(),
            dim=-1,
        )

        tprob = tlogp.exp()

        kl = (
            tprob
            * (
                tlogp
                - slogp
            )
        ).sum(
            dim=-1
        )

        out.append(
            kl.cpu()
        )

    del logits
    del ids
    del mask

    return out


def hint_gap(
    hint_topk,
    nohint_logp,
):
    import torch.nn.functional as F

    out = []

    for tk, nh in zip(
        hint_topk,
        nohint_logp,
    ):
        tvals, _ = tk

        tlogp = F.log_softmax(
            tvals.float(),
            dim=-1,
        )

        tprob = tlogp.exp()

        gap = (
            tprob
            * (
                tlogp
                - nh.float()
            )
        ).sum(
            dim=-1
        )

        out.append(
            gap.cpu()
        )

    return out


def pearson(xs, ys):
    if len(xs) < 2:
        return None

    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)

    dx = [
        x - mx
        for x in xs
    ]

    dy = [
        y - my
        for y in ys
    ]

    num = sum(
        a * b
        for a, b in zip(dx, dy)
    )

    den = math.sqrt(
        sum(a * a for a in dx)
        * sum(b * b for b in dy)
    )

    if den == 0:
        return None

    return num / den


def ranks(xs):
    order = sorted(
        range(len(xs)),
        key=lambda i: xs[i],
    )

    r = [0.0] * len(xs)

    i = 0

    while i < len(order):
        j = i + 1

        while (
            j < len(order)
            and xs[order[j]]
            == xs[order[i]]
        ):
            j += 1

        avg = (
            (i + 1 + j)
            / 2.0
        )

        for k in range(i, j):
            r[order[k]] = avg

        i = j

    return r


def random_share(
    values,
    count,
    repeats=500,
    seed=20260912,
):
    rng = random.Random(seed)

    total = sum(values)

    if total <= 0:
        return {
            "mean": None,
            "p05": None,
            "p95": None,
        }

    n = len(values)

    shares = []

    for _ in range(repeats):
        idx = rng.sample(
            range(n),
            count,
        )

        shares.append(
            sum(
                values[i]
                for i in idx
            )
            / total
        )

    shares.sort()

    return {
        "mean":
            statistics.fmean(shares),
        "p05":
            shares[int(
                0.05
                * (len(shares) - 1)
            )],
        "p95":
            shares[int(
                0.95
                * (len(shares) - 1)
            )],
    }


def summarize(path: Path):
    rows = read_jsonl(path)

    tokens = []

    lexical_hit_rows = 0

    for r in rows:
        if r["canonical_en_name_hit"]:
            lexical_hit_rows += 1

        for t in r["tokens"]:
            rec = dict(t)

            rec["job_id"] = r["job_id"]
            rec["src_term"] = r["src_term"]
            rec["en_name"] = r["en_name"]

            tokens.append(rec)

    kls = [
        float(t["opd_kl"])
        for t in tokens
    ]

    gaps_raw = [
        float(t["hint_gap"])
        for t in tokens
    ]

    min_gap = min(
        gaps_raw
    )

    require(
        min_gap > -1e-4,
        f"negative hint KL min={min_gap}",
    )

    gaps = [
        max(0.0, x)
        for x in gaps_raw
    ]

    total_kl = sum(kls)
    total_gap = sum(gaps)

    require(
        total_kl > 0,
        "zero total OPD KL",
    )

    order = sorted(
        range(len(tokens)),
        key=lambda i: gaps[i],
        reverse=True,
    )

    concentration = {}

    for frac in (
        0.10,
        0.20,
        0.50,
    ):
        count = max(
            1,
            math.ceil(
                frac
                * len(tokens)
            ),
        )

        idx = order[:count]

        kl_share = (
            sum(
                kls[i]
                for i in idx
            )
            / total_kl
        )

        gap_share = (
            sum(
                gaps[i]
                for i in idx
            )
            / total_gap
            if total_gap > 0
            else None
        )

        concentration[
            f"top_{int(frac*100)}pct_hint_sensitive"
        ] = {
            "token_count":
                count,
            "token_fraction":
                count / len(tokens),
            "opd_kl_mass_share":
                kl_share,
            "hint_gap_mass_share":
                gap_share,
            "random_equal_count_kl_share":
                random_share(
                    kls,
                    count,
                ),
        }

    thresholds = {}

    for th in (
        0.01,
        0.05,
        0.10,
        0.50,
        1.00,
    ):
        idx = [
            i
            for i, x in enumerate(gaps)
            if x >= th
        ]

        thresholds[
            f"hint_gap_ge_{th:g}"
        ] = {
            "tokens":
                len(idx),
            "token_fraction":
                len(idx) / len(tokens),
            "opd_kl_mass_share":
                (
                    sum(
                        kls[i]
                        for i in idx
                    )
                    / total_kl
                    if idx
                    else 0.0
                ),
        }

    top_examples = []

    for i in order[:100]:
        top_examples.append(
            tokens[i]
        )

    top_path = (
        OUT
        / "top100_hint_sensitive_tokens.jsonl"
    )

    tmp = top_path.with_name(
        top_path.name + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in top_examples:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp,
        top_path,
    )

    summary = {
        "status":
            "PASS",
        "scientific_class":
            "DIAGNOSTIC ONLY / A4 KL SIGNAL LOCALIZATION",
        "domain":
            "chemistry",
        "student_state":
            "C0",
        "rollout_epoch_seed_key":
            1,
        "rows":
            len(rows),
        "tokens":
            len(tokens),
        "canonical_en_name_hit_rows":
            lexical_hit_rows,
        "canonical_en_name_hit_rate":
            lexical_hit_rows / len(rows),
        "opd_kl": {
            "mean":
                statistics.fmean(kls),
            "sum":
                total_kl,
        },
        "hint_gap": {
            "mean":
                statistics.fmean(gaps),
            "sum":
                total_gap,
            "max":
                max(gaps),
        },
        "token_level_correlation": {
            "pearson":
                pearson(
                    gaps,
                    kls,
                ),
            "spearman":
                pearson(
                    ranks(gaps),
                    ranks(kls),
                ),
        },
        "concentration":
            concentration,
        "thresholds":
            thresholds,
        "definition": {
            "opd_kl":
                "KL(q_teacher_hint_top32 || p_student_full)",
            "hint_gap":
                "KL(q_teacher_hint_top32 || p_teacher_nohint_full)",
        },
    }

    atomic_json(
        OUT / "summary.json",
        summary,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    print(
        "FINAL_RESULT=A4_CHEMISTRY_KL_LOCALIZATION_PASS",
        flush=True,
    )


def run(args):
    import torch
    import torch_npu  # noqa: F401

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    h5 = load_source()

    rows = load_rows(
        args.n
    )

    result_path = (
        OUT
        / "rowwise_token_signals.jsonl"
    )

    done = existing_rows(
        result_path
    )

    pending = [
        r
        for r in rows
        if int(r["job_id"])
        not in done
    ]

    print(
        f"TARGET_ROWS={len(rows)} "
        f"RESUME_DONE={len(rows)-len(pending)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    if pending:
        sdev = "npu:0"
        tdev = "npu:1"

        torch.npu.set_device(
            sdev
        )

        st = AutoTokenizer.from_pretrained(
            C0
        )

        tt = AutoTokenizer.from_pretrained(
            TEACHER
        )

        require(
            h5.tokenizer_vocab_hash(st)
            == h5.tokenizer_vocab_hash(tt),
            "tokenizer vocab mismatch",
        )

        student = (
            AutoModelForCausalLM
            .from_pretrained(
                C0,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .to(sdev)
            .eval()
        )

        teacher = (
            AutoModelForCausalLM
            .from_pretrained(
                TEACHER,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .to(tdev)
            .eval()
        )

        student.config.use_cache = False
        teacher.config.use_cache = False

        for p in teacher.parameters():
            p.requires_grad_(False)

        started = time.time()
        initial = (
            len(rows)
            - len(pending)
        )

        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):
            batch_rows = pending[
                start:
                start + args.batch_size
            ]

            responses = (
                h5.generate_student_responses(
                    student,
                    st,
                    batch_rows,
                    sdev,
                    epoch=1,
                )
            )

            hint_topk = (
                h5.teacher_topk_for_batch(
                    teacher,
                    tt,
                    batch_rows,
                    responses,
                    tdev,
                )
            )

            nohint = (
                teacher_nohint_logp_on_hint_support(
                    teacher,
                    tt,
                    h5,
                    batch_rows,
                    responses,
                    hint_topk,
                    tdev,
                )
            )

            student_kls = (
                student_token_kl(
                    student,
                    st,
                    h5,
                    batch_rows,
                    responses,
                    hint_topk,
                    sdev,
                )
            )

            gaps = hint_gap(
                hint_topk,
                nohint,
            )

            for (
                row,
                resp,
                kls,
                hgs,
            ) in zip(
                batch_rows,
                responses,
                student_kls,
                gaps,
            ):
                require(
                    len(resp)
                    == len(kls)
                    == len(hgs),
                    (
                        "token alignment mismatch "
                        f"job={row['job_id']}"
                    ),
                )

                response_text = st.decode(
                    resp,
                    skip_special_tokens=True,
                )

                en_name = (
                    row["lexical_record"]
                    ["en_name"]
                    .strip()
                )

                token_rows = []

                for pos, (
                    token_id,
                    kl,
                    hg,
                ) in enumerate(
                    zip(
                        resp,
                        kls.tolist(),
                        hgs.tolist(),
                    )
                ):
                    token_rows.append(
                        {
                            "position":
                                pos,
                            "token_id":
                                int(token_id),
                            "token_text":
                                st.decode(
                                    [int(token_id)],
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False,
                                ),
                            "opd_kl":
                                float(kl),
                            "hint_gap":
                                float(hg),
                            "is_eos":
                                (
                                    int(token_id)
                                    == st.eos_token_id
                                ),
                        }
                    )

                rec = {
                    "status":
                        "PASS",
                    "domain":
                        "chemistry",
                    "student_state":
                        "C0",
                    "rollout_epoch":
                        1,
                    "job_id":
                        int(row["job_id"]),
                    "src_term":
                        row["src_term"],
                    "en_name":
                        en_name,
                    "src_text":
                        row["src_text"],
                    "response_text":
                        response_text,
                    "canonical_en_name_hit":
                        (
                            en_name.casefold()
                            in response_text.casefold()
                        ),
                    "response_token_count":
                        len(resp),
                    "tokens":
                        token_rows,
                }

                append_jsonl(
                    result_path,
                    rec,
                )

            completed = (
                initial
                + min(
                    start
                    + len(batch_rows),
                    len(pending),
                )
            )

            elapsed = max(
                time.time()
                - started,
                1e-9,
            )

            fresh = (
                completed
                - initial
            )

            rate = (
                fresh / elapsed
                if fresh
                else 0
            )

            eta = (
                (len(rows) - completed)
                / rate
                if rate
                else None
            )

            print(
                f"progress={completed}/{len(rows)} "
                f"rate={rate:.3f} rows/s "
                f"ETA="
                + (
                    "?"
                    if eta is None
                    else f"{eta/60:.1f}m"
                ),
                flush=True,
            )

    final_rows = existing_rows(
        result_path
    )

    selected_ids = {
        int(r["job_id"])
        for r in rows
    }

    final_selected = [
        r
        for jid, r in final_rows.items()
        if jid in selected_ids
    ]

    require(
        len(final_selected)
        == len(rows),
        (
            f"final selected rows="
            f"{len(final_selected)} "
            f"expected={len(rows)}"
        ),
    )

    if args.n >= 256:
        summarize(
            result_path
        )
    else:
        print(
            "FINAL_RESULT=A4_SMOKE_PASS",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    args = ap.parse_args()

    require(
        1 <= args.n <= 1000,
        f"bad n={args.n}",
    )

    run(args)


if __name__ == "__main__":
    main()
