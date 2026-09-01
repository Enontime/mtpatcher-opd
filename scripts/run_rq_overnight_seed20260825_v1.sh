#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
SEED="20260825"

TRAIN_FULL="$DATA_ROOT/$EXP/pe_k1_clean3732.jsonl"
TRAIN_MATCHED="$DATA_ROOT/$EXP/rq_ecropd_matched512_v1.jsonl"
SOURCE_MAP="$DATA_ROOT/$EXP/rq_ecropd_source_only6565_v1.jsonl"

STUDENT="$RUN_ROOT/$EXP/pe_k1_sft3732_b4ga4/epoch3"
TEACHER="$MODEL_ROOT/Qwen3-8B"

FULL_EC_SCRIPT="$ROOT/scripts/mtpatcher_ecropd_full3732_train_rngpaired_v1.py"
FULL_VAN_SCRIPT="$ROOT/scripts/mtpatcher_ecropd_full3732_vanilla_fkl_rngpaired_v1.py"

MATCH_EC_SCRIPT="$ROOT/scripts/mtpatcher_ecropd_matched512_train_v3_rngpaired.py"
MATCH_VAN_SCRIPT="$ROOT/scripts/mtpatcher_ecropd_matched512_vanilla_fkl_rngpaired_v1.py"

EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

EVAL_DATA="$DATA_ROOT/pilot_v2_qwen3_06b"

FULL_EC_OUT="$RUN_ROOT/$EXP/rq_full3732_ecropd_rngpaired_seed${SEED}_v1"
FULL_VAN_OUT="$RUN_ROOT/$EXP/rq_full3732_vanilla_rngpaired_seed${SEED}_v1"

MATCH_EC_OUT="$RUN_ROOT/$EXP/rq_matched512_ecropd_rngpaired_seed${SEED}_v1"
MATCH_VAN_OUT="$RUN_ROOT/$EXP/rq_matched512_vanilla_rngpaired_seed${SEED}_v1"

FULL_EC_EVAL="$RUN_ROOT/$EXP/rq_full3732_ecropd_rngpaired_seed${SEED}_eval_v1"
FULL_VAN_EVAL="$RUN_ROOT/$EXP/rq_full3732_vanilla_rngpaired_seed${SEED}_eval_v1"

MATCH_EC_EVAL="$RUN_ROOT/$EXP/rq_matched512_ecropd_rngpaired_seed${SEED}_eval_v1"
MATCH_VAN_EVAL="$RUN_ROOT/$EXP/rq_matched512_vanilla_rngpaired_seed${SEED}_eval_v1"

ANALYSIS="$RUN_ROOT/$EXP/rq_overnight_seed${SEED}_analysis_v1"

FULL_EC_LOG="${LOGS:-/workspace/mtpatcher/logs}/rq_full3732_ecropd_rngpaired_seed${SEED}_v1.log"
FULL_VAN_LOG="${LOGS:-/workspace/mtpatcher/logs}/rq_full3732_vanilla_rngpaired_seed${SEED}_v1.log"

MATCH_EC_LOG="${LOGS:-/workspace/mtpatcher/logs}/rq_matched512_ecropd_rngpaired_seed${SEED}_v1.log"
MATCH_VAN_LOG="${LOGS:-/workspace/mtpatcher/logs}/rq_matched512_vanilla_rngpaired_seed${SEED}_v1.log"

mkdir -p "$ANALYSIS"

echo "============================================================"
echo "OVERNIGHT RQ EXPLORATION"
echo "SEED=$SEED"
echo "START=$(date -Is)"
echo "============================================================"

echo
echo "===== PRECHECK HASHES ====="

sha256sum \
  "$FULL_EC_SCRIPT" \
  "$FULL_VAN_SCRIPT" \
  "$MATCH_EC_SCRIPT" \
  "$MATCH_VAN_SCRIPT" \
  "$TRAIN_FULL" \
  "$TRAIN_MATCHED"

EC_FULL_HASH="$(sha256sum "$FULL_EC_SCRIPT" | awk '{print $1}')"
VAN_FULL_HASH="$(sha256sum "$FULL_VAN_SCRIPT" | awk '{print $1}')"
DATA_HASH="$(sha256sum "$TRAIN_FULL" | awk '{print $1}')"

if [[ "$EC_FULL_HASH" != "b073ac03e57be47aeef1cc541e23f1657d5672b6ffa26607bae1df2948e51768" ]]; then
    echo "FULL_EC_HASH_MISMATCH"
    false
fi

if [[ "$VAN_FULL_HASH" != "a9bd6a2585bd132946b4c1aae3970eb51cb1985601f9d7414d3c4143823f9eac" ]]; then
    echo "FULL_VAN_HASH_MISMATCH"
    false
fi

if [[ "$DATA_HASH" != "5265e18a329f51d79d320a1a999f235a2e69c20e2428660173152711083c1619" ]]; then
    echo "FULL_DATA_HASH_MISMATCH"
    false
fi

echo "FROZEN_HASHES_PASS"

echo
echo "===== COLLISION GUARD ====="

for P in \
  "$FULL_EC_OUT" \
  "$FULL_VAN_OUT" \
  "$MATCH_EC_OUT" \
  "$MATCH_VAN_OUT" \
  "$FULL_EC_EVAL" \
  "$FULL_VAN_EVAL" \
  "$MATCH_EC_EVAL" \
  "$MATCH_VAN_EVAL"
do
    if [[ -e "$P" ]]; then
        echo "OUTPUT_ALREADY_EXISTS=$P"
        false
    fi
done

if pgrep -af 'mtpatcher_ecropd_full3732_train_rngpaired_v1.py' >/dev/null 2>&1; then
    echo "ACTIVE_FULL_EC_FOUND"
    pgrep -af 'mtpatcher_ecropd_full3732_train_rngpaired_v1.py'
    false
fi

if pgrep -af 'mtpatcher_ecropd_full3732_vanilla_fkl_rngpaired_v1.py' >/dev/null 2>&1; then
    echo "ACTIVE_FULL_VANILLA_FOUND"
    pgrep -af 'mtpatcher_ecropd_full3732_vanilla_fkl_rngpaired_v1.py'
    false
fi

echo "COLLISION_GUARD_PASS"

cat > "$ANALYSIS/literature_rationale.txt" <<'TXT'
Overnight questions:

Q1 Robustness:
Does the full3732 EC-vs-Vanilla signal survive a second pre-specified
paired RNG seed?

Q2 Scale x seed:
Under the same new seed, does the direction agree between matched512
and full3732?

Q3 Mechanism concentration:
Does EC help disproportionately where:
(a) Vanilla translation quality is low?
(b) the stronger Teacher has larger quality headroom over Vanilla?

Literature motivation:
- ImitKD / GKD: learner-visited states matter.
- Selective KD for NMT / MT-PATCHER: teacher knowledge is not uniformly useful.
- Rethinking OPD 2026: successful OPD requires genuinely new teacher capability.
- TIP / Teachability work: disagreement/supervision usefulness is non-uniform.

Important:
These are robustness and concentration diagnostics.
They do not isolate localization from correction conditioning.
No new EA-OPD implementation is introduced overnight.
TXT

echo
echo "============================================================"
echo "A. FULL3732 EC / SECOND SEED"
echo "============================================================"
echo "START=$(date -Is)"

torchrun \
  --standalone \
  --nproc_per_node=16 \
  "$FULL_EC_SCRIPT" \
  --student "$STUDENT" \
  --teacher "$TEACHER" \
  --train "$TRAIN_FULL" \
  --source-map "$SOURCE_MAP" \
  --output-dir "$FULL_EC_OUT" \
  --expected-rows 3732 \
  --epochs 3 \
  --lr 1e-6 \
  --max-prompt-length 512 \
  --max-new-tokens 256 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --seed "$SEED" \
  > "$FULL_EC_LOG" 2>&1

grep -q \
  'MTPATCHER_EC_ROPD_FULL3732_RNGPAIRED_PASS' \
  "$FULL_EC_LOG"

echo "FULL3732_EC_SECOND_SEED_PASS"
echo "END=$(date -Is)"

echo
echo "============================================================"
echo "B. FULL3732 VANILLA / SAME SECOND SEED"
echo "============================================================"
echo "START=$(date -Is)"

torchrun \
  --standalone \
  --nproc_per_node=16 \
  "$FULL_VAN_SCRIPT" \
  --student "$STUDENT" \
  --teacher "$TEACHER" \
  --train "$TRAIN_FULL" \
  --output-dir "$FULL_VAN_OUT" \
  --expected-rows 3732 \
  --epochs 3 \
  --lr 1e-6 \
  --max-prompt-length 512 \
  --max-new-tokens 256 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --seed "$SEED" \
  > "$FULL_VAN_LOG" 2>&1

grep -q \
  'MTPATCHER_V3_TORCHNPU_OPD_TRAINING_PASS' \
  "$FULL_VAN_LOG"

echo "FULL3732_VANILLA_SECOND_SEED_PASS"
echo "END=$(date -Is)"

evaluate_arm () {
    ARM="$1"
    MODEL_ROOT_PATH="$2"
    OUT_ROOT_PATH="$3"
    EPOCHS="$4"

    for E in $EPOCHS
    do
        MODEL="$MODEL_ROOT_PATH/epoch${E}"
        OUT="$OUT_ROOT_PATH/epoch${E}"

        mkdir -p "$OUT"

        for SPEC in \
          "wmt24:$EVAL_DATA/wmt24_zh_en998.jsonl" \
          "flores:$EVAL_DATA/flores_zh_en1012.jsonl" \
          "challenge:$EVAL_DATA/challenge_zh_en197.jsonl"
        do
            NAME="${SPEC%%:*}"
            INPUT="${SPEC#*:}"

            echo
            echo "EVAL ARM=$ARM EPOCH=$E DATASET=$NAME"

            python "$EVAL" \
              --model "$MODEL" \
              --input "$INPUT" \
              --output "$OUT/${NAME}.jsonl" \
              --method "${ARM}_seed${SEED}_e${E}_${NAME}" \
              --batch-size 16 \
              --max-new-tokens 256 \
              --attn-implementation sdpa

            python "$SCORE" \
              --input "$OUT/${NAME}.jsonl" \
              --output "$OUT/${NAME}_metrics.json"
        done
    done
}

echo
echo "============================================================"
echo "C. FULL3732 FIXED EVALUATION"
echo "============================================================"

evaluate_arm \
  "full3732_ec_rngpaired" \
  "$FULL_EC_OUT" \
  "$FULL_EC_EVAL" \
  "1 2 3"

evaluate_arm \
  "full3732_vanilla_rngpaired" \
  "$FULL_VAN_OUT" \
  "$FULL_VAN_EVAL" \
  "1 2 3"

echo
echo "============================================================"
echo "D. MATCHED512 / SAME NEW SEED"
echo "============================================================"

torchrun \
  --standalone \
  --nproc_per_node=16 \
  "$MATCH_EC_SCRIPT" \
  --student "$STUDENT" \
  --teacher "$TEACHER" \
  --train "$TRAIN_MATCHED" \
  --source-map "$SOURCE_MAP" \
  --output-dir "$MATCH_EC_OUT" \
  --expected-rows 512 \
  --epochs 3 \
  --lr 1e-6 \
  --max-prompt-length 512 \
  --max-new-tokens 256 \
  --feedback-max-prompt-tokens 1536 \
  --feedback-max-new-tokens 768 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --seed "$SEED" \
  > "$MATCH_EC_LOG" 2>&1

grep -q \
  'MTPATCHER_EC_ROPD_MATCHED512_RNGPAIRED_PASS' \
  "$MATCH_EC_LOG"

echo "MATCHED512_EC_SECOND_SEED_PASS"

torchrun \
  --standalone \
  --nproc_per_node=16 \
  "$MATCH_VAN_SCRIPT" \
  --student "$STUDENT" \
  --teacher "$TEACHER" \
  --train "$TRAIN_MATCHED" \
  --output-dir "$MATCH_VAN_OUT" \
  --expected-rows 512 \
  --epochs 3 \
  --lr 1e-6 \
  --max-prompt-length 512 \
  --max-new-tokens 256 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --seed "$SEED" \
  > "$MATCH_VAN_LOG" 2>&1

grep -q \
  'MTPATCHER_V3_TORCHNPU_OPD_TRAINING_PASS' \
  "$MATCH_VAN_LOG"

echo "MATCHED512_VANILLA_SECOND_SEED_PASS"

evaluate_arm \
  "matched512_ec_rngpaired" \
  "$MATCH_EC_OUT" \
  "$MATCH_EC_EVAL" \
  "3"

evaluate_arm \
  "matched512_vanilla_rngpaired" \
  "$MATCH_VAN_OUT" \
  "$MATCH_VAN_EVAL" \
  "3"

export EXP SEED ANALYSIS
export FULL_EC_EVAL FULL_VAN_EVAL
export MATCH_EC_EVAL MATCH_VAN_EVAL
export RUN_ROOT

echo
echo "============================================================"
echo "E. CROSS-SEED / CROSS-BUDGET SUMMARY"
echo "============================================================"

python - <<'PY' | tee "$ANALYSIS/main_summary.txt"
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_ROOT"])
exp = os.environ["EXP"]

datasets = ["wmt24", "flores", "challenge"]

systems = {
    "full_seed20260824": (
        run / exp / "rq_full3732_vanilla_rngpaired_eval_v1",
        run / exp / "rq_full3732_ecropd_rngpaired_eval_v1",
        [1, 2, 3],
    ),
    "full_seed20260825": (
        Path(os.environ["FULL_VAN_EVAL"]),
        Path(os.environ["FULL_EC_EVAL"]),
        [1, 2, 3],
    ),
    "matched_seed20260825": (
        Path(os.environ["MATCH_VAN_EVAL"]),
        Path(os.environ["MATCH_EC_EVAL"]),
        [3],
    ),
}

old_mv = run / exp / "rq_matched512_vanilla_rngpaired_eval_v1"
old_me = run / exp / "rq_matched512_ecropd_rngpaired_eval_v1"

if old_mv.exists() and old_me.exists():
    systems["matched_seed20260824"] = (
        old_mv,
        old_me,
        [3],
    )

def read(root, epoch, ds):
    p = root / f"epoch{epoch}" / f"{ds}_metrics.json"
    x = json.loads(p.read_text(encoding="utf-8"))
    return float(x["BLEU"]), float(x["chrF"])

print("name epoch van_bleu ec_bleu d_bleu van_chrf ec_chrf d_chrf")

e3_by_seed = {}

for name, (vroot, eroot, epochs) in systems.items():
    for e in epochs:
        vals = []
        for ds in datasets:
            vb, vc = read(vroot, e, ds)
            eb, ec = read(eroot, e, ds)
            vals.append((vb, vc, eb, ec))

        vb = sum(x[0] for x in vals) / 3
        vc = sum(x[1] for x in vals) / 3
        eb = sum(x[2] for x in vals) / 3
        ec = sum(x[3] for x in vals) / 3

        print(
            name,
            e,
            f"{vb:.6f}",
            f"{eb:.6f}",
            f"{eb-vb:+.6f}",
            f"{vc:.6f}",
            f"{ec:.6f}",
            f"{ec-vc:+.6f}",
        )

        if e == 3 and name.startswith("full_"):
            e3_by_seed[name] = (eb-vb, ec-vc)

print()
print("FULL_E3_DATASET_DELTAS")

for name in sorted(e3_by_seed):
    vroot, eroot, _ = systems[name]
    print(name)

    for ds in datasets:
        vb, vc = read(vroot, 3, ds)
        eb, ec = read(eroot, 3, ds)

        print(
            f"  {ds}: "
            f"dBLEU={eb-vb:+.6f} "
            f"dchrF={ec-vc:+.6f}"
        )

if len(e3_by_seed) >= 2:
    mean_b = sum(x[0] for x in e3_by_seed.values()) / len(e3_by_seed)
    mean_c = sum(x[1] for x in e3_by_seed.values()) / len(e3_by_seed)

    print()
    print(
        "TWO_SEED_FULL_E3_MEAN "
        f"dBLEU={mean_b:+.6f} "
        f"dchrF={mean_c:+.6f}"
    )

print()
print("CROSS_SEED_SUMMARY_PASS")
PY

echo
echo "============================================================"
echo "F. OFFICIAL SACREBLEU PAIRED TESTS"
echo "============================================================"

mkdir -p "$ANALYSIS/significance"

python - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_ROOT"])
exp = os.environ["EXP"]
analysis = Path(os.environ["ANALYSIS"])

pairs = {
    "seed20260824": (
        run / exp / "rq_full3732_vanilla_rngpaired_eval_v1/epoch3",
        run / exp / "rq_full3732_ecropd_rngpaired_eval_v1/epoch3",
    ),
    "seed20260825": (
        Path(os.environ["FULL_VAN_EVAL"]) / "epoch3",
        Path(os.environ["FULL_EC_EVAL"]) / "epoch3",
    ),
}

for seed, (vroot, eroot) in pairs.items():
    for ds in ["wmt24", "flores", "challenge"]:
        def load(p):
            rows = [
                json.loads(x)
                for x in p.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            rows.sort(key=lambda x: int(x["index"]))
            return rows

        v = load(vroot / f"{ds}.jsonl")
        e = load(eroot / f"{ds}.jsonl")

        assert len(v) == len(e)

        for a, b in zip(v, e):
            assert a["index"] == b["index"]
            assert a["reference"] == b["reference"]
            assert a["source"] == b["source"]

        out = analysis / "significance" / seed / ds
        out.mkdir(parents=True, exist_ok=True)

        (out / "ref.txt").write_text(
            "\n".join(x["reference"] for x in v) + "\n",
            encoding="utf-8",
        )
        (out / "vanilla.txt").write_text(
            "\n".join(x["student_translation"] for x in v) + "\n",
            encoding="utf-8",
        )
        (out / "ec.txt").write_text(
            "\n".join(x["student_translation"] for x in e) + "\n",
            encoding="utf-8",
        )

print("SIGNIFICANCE_INPUTS_PASS")
PY

export SACREBLEU_SEED=20260824

for S in seed20260824 seed20260825
do
    for D in wmt24 flores challenge
    do
        SD="$ANALYSIS/significance/$S/$D"

        echo
        echo "PAIRED_BS $S $D"

        sacrebleu \
          "$SD/ref.txt" \
          -i "$SD/vanilla.txt" "$SD/ec.txt" \
          -m bleu chrf \
          --paired-bs \
          --paired-bs-n 1000 \
          --paired-jobs 1 \
          -f text \
          | tee "$SD/paired_bs.txt"

        echo
        echo "PAIRED_AR $S $D"

        sacrebleu \
          "$SD/ref.txt" \
          -i "$SD/vanilla.txt" "$SD/ec.txt" \
          -m bleu chrf \
          --paired-ar \
          --paired-ar-n 5000 \
          --paired-jobs 1 \
          -f text \
          | tee "$SD/paired_ar.txt"
    done
done

echo
echo "============================================================"
echo "G. DIFFICULTY / TEACHER-HEADROOM STRATIFICATION"
echo "============================================================"

export QGATE="$DATA_ROOT/$EXP/patcher_quality_gate_v1/quality_gate_materialized.jsonl"

python - <<'PY' | tee "$ANALYSIS/stratified_diagnosis.txt"
import json
import os
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

run = Path(os.environ["RUN_ROOT"])
exp = os.environ["EXP"]

bleu = BLEU()
chrf = CHRF()

seed_roots = {
    "seed20260824": (
        run / exp / "rq_full3732_vanilla_rngpaired_eval_v1/epoch3",
        run / exp / "rq_full3732_ecropd_rngpaired_eval_v1/epoch3",
    ),
    "seed20260825": (
        Path(os.environ["FULL_VAN_EVAL"]) / "epoch3",
        Path(os.environ["FULL_EC_EVAL"]) / "epoch3",
    ),
}

datasets = ["wmt24", "flores", "challenge"]

def load(path):
    rows = [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    rows.sort(key=lambda x: int(x["index"]))
    return rows

def sent_chrf(h, r):
    return chrf.sentence_score(h, [r]).score

def score_group(records):
    vh = [r["van"] for r in records]
    eh = [r["ec"] for r in records]
    refs = [r["ref"] for r in records]

    vb = bleu.corpus_score(vh, [refs]).score
    eb = bleu.corpus_score(eh, [refs]).score

    vc = chrf.corpus_score(vh, [refs]).score
    ec = chrf.corpus_score(eh, [refs]).score

    changed = sum(r["van"] != r["ec"] for r in records) / len(records)

    return eb-vb, ec-vc, changed

def quartiles(records, key):
    ordered = sorted(records, key=lambda r: r[key])
    n = len(ordered)

    result = []

    for q in range(4):
        lo = n * q // 4
        hi = n * (q + 1) // 4
        result.append(ordered[lo:hi])

    return result

qgate_path = Path(os.environ["QGATE"])
teacher_map = {}

if qgate_path.exists():
    for r in load(qgate_path):
        src = r.get("source")
        ref = r.get("reference")
        teacher = r.get("direct_teacher")

        if (
            isinstance(src, str)
            and isinstance(ref, str)
            and isinstance(teacher, str)
            and teacher.strip()
        ):
            key = (src, ref)
            if key not in teacher_map:
                teacher_map[key] = teacher

print("DIFFICULTY PROXY:")
print("Vanilla sentence-chrF rank only defines strata.")
print("Outcome inside each stratum remains corpus BLEU / corpus chrF.")
print()
print("HEADROOM PROXY:")
print("Teacher sentence-chrF - Vanilla sentence-chrF only defines strata.")
print("This is an exploratory proxy for teacher novelty, not proof of teachability.")
print()

for seed, (vroot, eroot) in seed_roots.items():

    print("=" * 72)
    print(seed)
    print("=" * 72)

    diff_macro = [[] for _ in range(4)]
    head_macro = [[] for _ in range(4)]

    for ds in datasets:
        v = load(vroot / f"{ds}.jsonl")
        e = load(eroot / f"{ds}.jsonl")

        records = []

        for a, b in zip(v, e):
            assert a["index"] == b["index"]
            assert a["source"] == b["source"]
            assert a["reference"] == b["reference"]

            ref = a["reference"]
            van = a["student_translation"]
            eco = b["student_translation"]

            r = {
                "source": a["source"],
                "ref": ref,
                "van": van,
                "ec": eco,
                "van_quality": sent_chrf(van, ref),
            }

            teacher = teacher_map.get(
                (a["source"], ref)
            )

            if teacher is not None:
                r["teacher_headroom"] = (
                    sent_chrf(teacher, ref)
                    - r["van_quality"]
                )

            records.append(r)

        print()
        print(ds)

        print("  DIFFICULTY_QUARTILES")
        dq = quartiles(records, "van_quality")

        for qi, group in enumerate(dq, 1):
            db, dc, cr = score_group(group)
            diff_macro[qi-1].append((db, dc))

            print(
                f"    Q{qi} "
                f"n={len(group)} "
                f"dBLEU={db:+.6f} "
                f"dchrF={dc:+.6f} "
                f"changed={cr:.3f}"
            )

        with_teacher = [
            r for r in records
            if "teacher_headroom" in r
        ]

        coverage = len(with_teacher) / len(records)

        print(
            f"  TEACHER_HEADROOM_COVERAGE={coverage:.3f}"
        )

        if coverage >= 0.90:
            print("  HEADROOM_QUARTILES")

            hq = quartiles(
                with_teacher,
                "teacher_headroom",
            )

            for qi, group in enumerate(hq, 1):
                db, dc, cr = score_group(group)
                head_macro[qi-1].append((db, dc))

                print(
                    f"    Q{qi} "
                    f"n={len(group)} "
                    f"dBLEU={db:+.6f} "
                    f"dchrF={dc:+.6f} "
                    f"changed={cr:.3f}"
                )

    print()
    print("  DIFFICULTY_MACRO_BY_QUARTILE")

    for qi, xs in enumerate(diff_macro, 1):
        if xs:
            print(
                f"    Q{qi} "
                f"dBLEU={sum(x[0] for x in xs)/len(xs):+.6f} "
                f"dchrF={sum(x[1] for x in xs)/len(xs):+.6f}"
            )

    print()
    print("  HEADROOM_MACRO_BY_QUARTILE")

    for qi, xs in enumerate(head_macro, 1):
        if xs:
            print(
                f"    Q{qi} "
                f"dBLEU={sum(x[0] for x in xs)/len(xs):+.6f} "
                f"dchrF={sum(x[1] for x in xs)/len(xs):+.6f}"
            )

    print()

print("STRATIFIED_DIAGNOSIS_PASS")
PY

echo
echo "============================================================"
echo "OVERNIGHT COMPLETE"
echo "END=$(date -Is)"
echo "ANALYSIS=$ANALYSIS"
echo "============================================================"
