#!/usr/bin/env bash
# Only engineering smoke. NO formal training. NO git operations.
set -euo pipefail
RUN="${1:-}"
if [[ -z "$RUN" || ! -f "$RUN/run_contract.json" || ! -f "$RUN/overrides.txt" ]]; then
  echo 'Usage: bash recipes/opd/run_chemistry_onpolicy_smoke_v1.sh /absolute/prepared/run' >&2
  exit 2
fi
ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
VERL="$ROOT/repo/verl-v0.9.0"
PY="$ROOT/envs/verl-v0.9.0-a3/bin/python"
export PYTHONPATH="$PROJECT/scripts/opd:$PROJECT/scripts/targeted:$PROJECT:$VERL:${PYTHONPATH:-}"
export PATH="$ROOT/envs/verl-v0.9.0-a3/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
"$PY" - "$RUN" "$PROJECT" "$VERL" <<'PY_GATE'
import json,sys,subprocess
from pathlib import Path
run,proj,verl=map(Path,sys.argv[1:])
c=json.loads((run/'run_contract.json').read_text())
assert c['authorization']=='ONE_STEP_ENGINEERING_SMOKE_ONLY'
assert c['mode']=='smoke' and c['population']==16
assert c.get('RUN_CLASS') == 'smoke', 'RUN_CLASS violation'
assert run.resolve().parent == (Path('/workspace/mtpatcher/runs/smoke/targeted')).resolve(), 'SMOKE_PATH_VIOLATION'
assert any(x=='reward.custom_reward_function.path='+str(proj/'scripts/opd/chemistry_zero_reward_v1.py') for x in (run/'overrides.txt').read_text().splitlines()), 'REWARD_CONTRACT_VIOLATION'
assert subprocess.check_output(['git','-C',str(proj),'rev-parse','HEAD'],text=True).strip()==c['project_commit']
assert subprocess.check_output(['git','-C',str(verl),'rev-parse','HEAD'],text=True).strip()==c['verl_commit']
assert subprocess.check_output(['git','-C',str(proj),'status','--porcelain'],text=True)==c['project_dirty']
from importlib import import_module
for name in ('torch_npu','ray','tensorboard','omegaconf','pyarrow','transformers'):
    import_module(name)
print('ENV_AND_GIT=PASS')
PY_GATE
if pgrep -af '[p]ython.*main_ppo_sync_wrapper_v1.py' >/dev/null; then
  echo 'EXISTING_VERL_SYNC_TRAINER: refusing concurrent run' >&2
  exit 3
fi
# verl native TB writer uses a project-relative log_dir: redirect this unique
# smoke experiment into the corresponding run, not into a shared/formal TB view.
EXP="$("$PY" - "$RUN/overrides.txt" <<'PY_EXPNAME'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
vals=[line.split('=',1)[1] for line in lines if line.startswith('trainer.experiment_name=')]
assert len(vals)==1 and vals[0].startswith('smoke-chem-onpolicy-')
print(vals[0])
PY_EXPNAME
)"
TB_ROOT="$PROJECT/tensorboard_log/mtpatcher-chem-onpolicy-smoke"
TB_TARGET="$RUN/tensorboard/native"
TB_LINK="$TB_ROOT/$EXP"
mkdir -p "$TB_ROOT" "$TB_TARGET"
if [[ -e "$TB_LINK" || -L "$TB_LINK" ]]; then
  echo "TENSORBOARD_IDENTITY_COLLISION: $TB_LINK" >&2
  exit 4
fi
ln -s "$TB_TARGET" "$TB_LINK"
cd "$PROJECT"
mapfile -t OVERRIDES < "$RUN/overrides.txt"
: > "$RUN/run.log"
"$PY" "$PROJECT/scripts/targeted/main_ppo_sync_wrapper_v1.py" \
  "${OVERRIDES[@]}" >> "$RUN/run.log" 2>&1
"$PY" - "$RUN" <<'PY_END'
from pathlib import Path
import sys
p=Path(sys.argv[1]); files=list((p/'checkpoints').rglob('*'))
assert any(x.name.startswith('global_step_') for x in files), 'NO_VERL_CHECKPOINT'
assert (p/'chem_onpolicy_rollout_metrics.jsonl').exists(), 'NO_FRESH_ROLLOUT_PROOF'
assert list((p/'tensorboard'/'rollout').glob('events.out.tfevents.*')), 'NO_REAL_ROLLOUT_TENSORBOARD'
print('ENGINEERING_SMOKE_ARTIFACTS_PRESENT; inspect native actor KL + optimizer + event steps before scientific acceptance')
PY_END
