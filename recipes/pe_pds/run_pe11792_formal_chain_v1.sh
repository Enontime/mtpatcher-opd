#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"

cd "$PROJECT"

echo "============================================================"
echo "PE11792 FORMAL FRESH MATCHED CHAIN"
echo "============================================================"
echo "START_UTC=$(date -u +%FT%TZ)"

bash recipes/pe_pds/run_pe11792_sft_formal_v1.sh
SFT_RC=$?

echo
echo "SFT_RC=$SFT_RC"
echo "SFT_END_UTC=$(date -u +%FT%TZ)"

# OPD is an independent preregistered arm.
# Attempt it even if the SFT arm exits nonzero; its own preflight
# will block if the machine is not safe to use.
bash recipes/pe_pds/run_pe11792_opd_formal_v1.sh
OPD_RC=$?

echo
echo "OPD_RC=$OPD_RC"
echo "OPD_END_UTC=$(date -u +%FT%TZ)"

if [[ "$SFT_RC" -eq 0 && "$OPD_RC" -eq 0 ]]; then
    echo "PE11792_FORMAL_CHAIN=PASS"
    exit 0
fi

echo "PE11792_FORMAL_CHAIN=INCOMPLETE"
exit 1
