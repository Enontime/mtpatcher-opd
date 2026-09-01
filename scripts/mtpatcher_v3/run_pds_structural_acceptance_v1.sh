#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

BASE="$D/pds_smoke_analysis_clean_ablation_v3"

JOBS="$BASE/pds_jobs_analysis_clean1100_v2.jsonl"
RAW="$BASE/pds_raw_analysis_clean1100_v2.jsonl"

OLD_HIDDEN="$BASE/longq_semantic_audit_v1/longq_semantic_hidden_manifest_v1.jsonl"

OUT="$BASE/structural_acceptance_v1"

PY="$ROOT/scripts/mtpatcher_v3/build_pds_structural_acceptance_v1.py"

PASS="$OUT/PDS_STRUCTURAL_ACCEPTANCE_V1.PASS"
FAIL="$OUT/PDS_STRUCTURAL_ACCEPTANCE_V1.FAIL"

mkdir -p "$OUT"

rm -f "$PASS" "$FAIL"

START="$(date +%s)"

echo "======================================================================"
echo "PDS STRUCTURAL ACCEPTANCE V1"
echo "======================================================================"
echo "NO MODEL LOAD / NO GENERATION / NO TRAINING"
echo
echo "预计运行时长：1–5 秒；保守上限 30 秒"
echo "开始时间："
echo "  北京时间: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "GENERATOR = FROZEN"
echo "Q exact = diagnostic only"
echo

trap '
rc=$?
echo
echo "PDS_STRUCTURAL_ACCEPTANCE_V1_FAILED rc=$rc"
echo "time=$(date -Iseconds)"
touch "'"$FAIL"'"
' ERR


###############################################################################
# PREFLIGHT
###############################################################################

for F in \
    "$JOBS" \
    "$RAW" \
    "$OLD_HIDDEN" \
    "$PY"
do

    if [ ! -f "$F" ]; then
        echo "MISSING=$F"
        false
    fi

done

[ "$(wc -l < "$JOBS")" -eq 1100 ] || false
[ "$(wc -l < "$RAW")" -eq 1100 ] || false
[ "$(wc -l < "$OLD_HIDDEN")" -eq 60 ] || false

python -m py_compile "$PY"


###############################################################################
# BUILD
###############################################################################

python -u "$PY" \
    "$JOBS" \
    "$RAW" \
    "$OLD_HIDDEN" \
    "$OUT"


###############################################################################
# FREEZE
###############################################################################

sha256sum \
    "$JOBS" \
    "$RAW" \
    "$OLD_HIDDEN" \
    "$PY" \
    "$OUT/pds_structural_parse_all1100_v1.jsonl" \
    "$OUT/pds_structural_accepted_A_v1.jsonl" \
    "$OUT/pds_structural_rejected_v1.jsonl" \
    "$OUT/pds_structural_acceptance_report_v1.json" \
    "$OUT/accepted_set_semantic_blind100_v1.csv" \
    "$OUT/accepted_set_semantic_hidden100_v1.jsonl" \
    "$OUT/accepted_set_semantic_protocol_v1.txt" \
    > "$OUT/frozen_sha256_manifest_v1.txt"


touch "$PASS"
rm -f "$FAIL"

END="$(date +%s)"

echo
echo "======================================================================"
echo "PDS STRUCTURAL ACCEPTANCE V1 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$((END - START))"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "PASS=$PASS"
echo "======================================================================"
