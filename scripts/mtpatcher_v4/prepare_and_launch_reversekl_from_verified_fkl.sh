#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "======================================================================"
echo "MT-PATCHER V4 — BUILD REVERSE-KL OPD FROM VERIFIED FORWARD-KL PIPELINE"
date
echo "======================================================================"

###############################################################################
# STAGE 0 — STOP THE BROKEN V2 ONLY
###############################################################################

echo
echo "===== STAGE 0: STOP BROKEN V2 ====="

OLD_PIDS="$(pgrep -f 'train_pe_opd_reversekl_v2.py' || true)"

if [[ -n "$OLD_PIDS" ]]; then
    echo "Stopping old trainer PIDs:"
    echo "$OLD_PIDS"
    kill $OLD_PIDS || true
    sleep 3
else
    echo "No broken v2 trainer is running."
fi

OLD_MASTER_PIDS="$(pgrep -f 'run_pe_opd_reversekl_v2_oneclick.sh' || true)"

if [[ -n "$OLD_MASTER_PIDS" ]]; then
    echo "Stopping old master PIDs:"
    echo "$OLD_MASTER_PIDS"
    kill $OLD_MASTER_PIDS || true
    sleep 2
else
    echo "No broken v2 master is running."
fi


###############################################################################
# STAGE 1 — CLONE THE VERIFIED FORWARD-KL TRAINER
###############################################################################

echo
echo "===== STAGE 1: BUILD REVERSE-KL TRAINER ====="

SRC_TRAINER="$ROOT/scripts/mtpatcher_v3/train_opd_forwardkl_torchnpu.py"
DST_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py"

export SRC_TRAINER DST_TRAINER

python - <<'PY'
import os
import re
from pathlib import Path

src = Path(os.environ["SRC_TRAINER"])
dst = Path(os.environ["DST_TRAINER"])

if not src.exists():
    raise RuntimeError(
        f"Verified forward-KL trainer missing: {src}"
    )

s = src.read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# Sanity: the source trainer must be the already-verified implementation.
# ----------------------------------------------------------------------

required_markers = [
    "ON_POLICY = True",
    "D_KL(",
    "student.module.generate",
    "teacher_out = teacher(",
    "token_kl",
    "clip_grad_norm_",
    "DistributedSampler",
]

for marker in required_markers:
    if marker not in s:
        raise RuntimeError(
            f"Verified trainer marker missing: {marker}"
        )


# ----------------------------------------------------------------------
# Replace only the KL block.
#
# Original:
#
#   t_logp = log_softmax(T)
#   t_prob = exp(t_logp)
#   s_logp = log_softmax(S)
#   KL = sum t_prob * (t_logp - s_logp)
#
# New:
#
#   t_logp = log_softmax(T)
#   s_logp = log_softmax(S)
#   s_prob = exp(s_logp)
#   KL = sum s_prob * (s_logp - t_logp)
#
# Everything else remains identical.
# ----------------------------------------------------------------------

pattern = re.compile(
    r'''
                t_logp\s*=\s*F\.log_softmax\(
                    \s*t_logits\.float\(\),
                    \s*dim=-1,
                \)

                \s*t_prob\s*=\s*t_logp\.exp\(\)

            \s*s_logp\s*=\s*F\.log_softmax\(
                \s*s_logits\.float\(\),
                \s*dim=-1,
            \)

            .*?
            token_kl\s*=\s*\(
                \s*t_prob
                \s*\*\s*\(
                    \s*t_logp
                    \s*-\s*s_logp
                \s*\)
            \s*\)\.sum\(dim=-1\)

            \s*loss\s*=\s*token_kl\.mean\(\)
    ''',
    flags=re.VERBOSE | re.DOTALL,
)

replacement = '''
                t_logp = F.log_softmax(
                    t_logits.float(),
                    dim=-1,
                )

            s_logp = F.log_softmax(
                s_logits.float(),
                dim=-1,
            )

            # Exact full-vocabulary reverse KL:
            #
            # D_KL(P_student || P_teacher)
            #
            # Student probabilities are derived from float32 log-softmax.
            # This avoids the fp16 softmax/log instability that caused the
            # abandoned v2 implementation to produce NaN.
            s_prob = s_logp.exp()

            token_kl = (
                s_prob
                * (
                    s_logp
                    - t_logp
                )
            ).sum(dim=-1)

            loss = token_kl.mean()
'''

s2, count = pattern.subn(
    replacement,
    s,
    count=1,
)

if count != 1:
    raise RuntimeError(
        "Could not uniquely locate the verified forward-KL block. "
        f"replacement_count={count}"
    )

s = s2

# ----------------------------------------------------------------------
# Change diagnostic labels only.
# ----------------------------------------------------------------------

s = s.replace(
    "TORCH-NPU ON-POLICY FORWARD-KL",
    "TORCH-NPU ON-POLICY REVERSE-KL",
)

s = s.replace(
    '"Teacher || Student"',
    '"Student || Teacher"',
)

s = s.replace(
    "token_mean_forward_kl",
    "token_mean_reverse_kl",
)

s = s.replace(
    "MTPATCHER_V3_TORCHNPU_OPD_TRAINING_PASS",
    "MTPATCHER_V4_TORCHNPU_REVERSEKL_TRAINING_PASS",
)

# ----------------------------------------------------------------------
# Verify mathematically important pieces.
# ----------------------------------------------------------------------

checks = [
    "D_KL(P_student || P_teacher)",
    "s_prob = s_logp.exp()",
    "s_logp",
    "- t_logp",
    "student.module.generate",
    "ON_POLICY = True",
]

for marker in checks:
    if marker not in s:
        raise RuntimeError(
            f"Reverse-KL verification failed: {marker}"
        )

# Ensure the old forward probability block is gone.
if "t_prob = t_logp.exp()" in s:
    raise RuntimeError(
        "Old forward-KL teacher probability block still exists"
    )

dst.write_text(
    s,
    encoding="utf-8",
)

print("SOURCE =", src)
print("TARGET =", dst)
print("KL_BLOCK_REPLACEMENTS =", count)
print("REVERSE_KL_TRAINER_BUILD_PASS")
PY


###############################################################################
# STAGE 2 — PYTHON COMPILE + SOURCE AUDIT
###############################################################################

echo
echo "===== STAGE 2: TRAINER AUDIT ====="

python -m py_compile "$DST_TRAINER"

echo
echo "Important trainer lines:"
grep -nE \
'REVERSE-KL|Student \|\| Teacher|P_student \|\| P_teacher|s_prob =|token_mean_reverse_kl|ON_POLICY' \
"$DST_TRAINER"

echo
echo "TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 3 — CLONE THE VERIFIED ONE-CLICK PIPELINE
###############################################################################

echo
echo "===== STAGE 3: BUILD ONE-CLICK PIPELINE ====="

SRC_MASTER="$ROOT/scripts/mtpatcher_v3/run_opd_forwardkl_torchnpu_oneclick.sh"
DST_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_reversekl_torchnpu_oneclick.sh"

export SRC_MASTER DST_MASTER

python - <<'PY'
import os
from pathlib import Path

src = Path(os.environ["SRC_MASTER"])
dst = Path(os.environ["DST_MASTER"])

if not src.exists():
    raise RuntimeError(
        f"Verified forward-KL master missing: {src}"
    )

s = src.read_text(
    encoding="utf-8"
)

required = [
    'NAME="opd_torchnpu_fkl_pe3732_v1"',
    "train_opd_forwardkl_torchnpu.py",
    "--nproc_per_node=16",
    "pe_k1_clean3732.jsonl",
    "SeqKD-Selected3732",
]

for marker in required:
    if marker not in s:
        raise RuntimeError(
            f"Verified master marker missing: {marker}"
        )

# New experiment identity.
s = s.replace(
    'NAME="opd_torchnpu_fkl_pe3732_v1"',
    'NAME="opd_torchnpu_rkl_pe3732_v1"',
)

# Use the new trainer.
s = s.replace(
    '$ROOT/scripts/mtpatcher_v3/train_opd_forwardkl_torchnpu.py',
    '$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py',
)

# Avoid any stale rendezvous/port conflict.
s = s.replace(
    "--master_port=29631",
    "--master_port=29635",
)

# Evaluation method labels.
s = s.replace(
    "opd_torchnpu_fkl_pe3732_",
    "opd_torchnpu_rkl_pe3732_",
)

# Human-readable labels.
s = s.replace(
    "ON-POLICY FORWARD-KL",
    "ON-POLICY REVERSE-KL",
)

s = s.replace(
    "OPD-FKL-PE3732",
    "OPD-RKL-PE3732",
)

# Summary artifact filename.
s = s.replace(
    "opd_torchnpu_final_summary.json",
    "opd_torchnpu_reversekl_final_summary.json",
)

s = s.replace(
    "MTPATCHER_V3_TORCHNPU_OPD_ALL_PASS",
    "MTPATCHER_V4_TORCHNPU_REVERSEKL_ALL_PASS",
)

dst.write_text(
    s,
    encoding="utf-8",
)

print("SOURCE =", src)
print("TARGET =", dst)
print("REVERSE_KL_MASTER_BUILD_PASS")
PY

chmod +x "$DST_MASTER"

bash -n "$DST_MASTER"

echo
echo "Important master lines:"
grep -nE \
'NAME=|reversekl|REVERSE-KL|RKL|nproc_per_node|master_port|TRAINER=' \
"$DST_MASTER"

echo
echo "MASTER_SCRIPT_AUDIT_PASS"


###############################################################################
# STAGE 4 — FINAL PREFLIGHT
###############################################################################

echo
echo "===== STAGE 4: FINAL PREFLIGHT ====="

export DST_TRAINER DST_MASTER

python - <<'PY'
import os
from pathlib import Path

trainer = Path(
    os.environ["DST_TRAINER"]
)

master = Path(
    os.environ["DST_MASTER"]
)

t = trainer.read_text(
    encoding="utf-8"
)

m = master.read_text(
    encoding="utf-8"
)

checks = {
    "trainer_on_policy":
        "ON_POLICY = True" in t,

    "trainer_reverse_kl":
        "D_KL(P_student || P_teacher)" in t,

    "student_probability":
        "s_prob = s_logp.exp()" in t,

    "no_forward_t_prob":
        "t_prob = t_logp.exp()" not in t,

    "student_rollout":
        "student.module.generate" in t,

    "finite_loss_guard":
        "torch.isfinite(loss)" in t,

    "gradient_clip":
        "clip_grad_norm_" in t,

    "16_npu":
        "--nproc_per_node=16" in m,

    "pe_source_set":
        "pe_k1_clean3732.jsonl" in m,

    "reverse_trainer":
        "train_opd_reversekl_torchnpu.py" in m,
}

for k, v in checks.items():
    print(
        f"{k:25s} = {v}"
    )

if not all(checks.values()):
    raise RuntimeError(
        "Final reverse-KL preflight failed"
    )

print()
print("REVERSE_KL_FINAL_PREFLIGHT_PASS")
PY


###############################################################################
# STAGE 5 — LAUNCH WITH NOHUP + SETSID
###############################################################################

echo
echo "===== STAGE 5: LAUNCH ====="

LOG="$LOG_ROOT/mtpatcher_v3_full6565_20260823/opd_torchnpu_rkl_pe3732_v1.log"

mkdir -p "$(dirname "$LOG")"

# New experiment path/log, so no previous successful artifacts are overwritten.
nohup setsid bash "$DST_MASTER" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!

echo
echo "======================================================================"
echo "REVERSE-KL OPD STARTED"
echo "PID=$PID"
echo "LOG=$LOG"
echo "TRAINER=$DST_TRAINER"
echo "MASTER=$DST_MASTER"
echo "======================================================================"
