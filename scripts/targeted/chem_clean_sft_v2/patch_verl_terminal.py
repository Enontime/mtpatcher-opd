#!/usr/bin/env python3
"""Minimal, idempotent, fail-closed patch to the *native* Verl SFT source.
Backup is not hidden; git diff is the final provenance. Does not touch other Verl forks.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

SFT = Path('/workspace/mtpatcher/repo/verl-v0.9.0/verl/trainer/sft_trainer.py')
OLD = 'if is_last_step or (self.save_freq > 0 and is_save_step):'
MARKER = 'CHEM_SFT_V2_EPOCH_FINAL_SAVE'
NEW = '''# CHEM_SFT_V2_EPOCH_FINAL_SAVE: total_training_steps is the LR horizon;
                # the dataloader may end earlier because drop_last=True. Preserve
                # that horizon while saving the final completed optimizer update.
                is_epoch_terminal_step = (
                    epoch + 1 == self.config.trainer.total_epochs
                    and step_in_epoch + 1 == self.steps_per_epoch
                )
                if is_last_step or (self.save_freq > 0 and is_save_step) or is_epoch_terminal_step:'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['inspect','apply'])
    args = ap.parse_args()
    if not SFT.is_file():
        raise RuntimeError('NATIVE_VERL_SFT_NOT_FOUND')
    data = SFT.read_text(encoding='utf-8')
    print('SFT_FILE', SFT)
    print('SFT_SHA256', hashlib.sha256(data.encode()).hexdigest())
    if MARKER in data:
        if NEW not in data:
            raise RuntimeError('PARTIAL_OR_UNEXPECTED_PREVIOUS_PATCH')
        print('PATCH_ALREADY_PRESENT')
        return
    assert data.count(OLD) == 1, 'EXPECTED_CHECKPOINT_BRANCH_NOT_UNIQUE'
    assert 'for step_in_epoch, data in enumerate(' in data, 'UNKNOWN_SFT_LOOP'
    assert 'for epoch in range(' in data and 'self.steps_per_epoch' in data, 'UNKNOWN_EPOCH_SEMANTICS'
    assert 'self.ckpt_handler.save_checkpoint(step=global_step)' in data, 'UNKNOWN_CHECKPOINT_HANDLER'
    print('PATCH_COMPATIBLE')
    if args.mode == 'inspect':
        return
    backup_dir = Path('/workspace/mtpatcher/smoke/chem_clean_sft_v2/patch_backups')
    backup = backup_dir / ('sft_trainer_' + hashlib.sha256(data.encode()).hexdigest()[:16] + '.pre_patch.bak')
    assert not backup.exists(), f'BACKUP_ALREADY_PRESENT {backup}'
    # Only patch a clean SFT source unless explicitly reviewed first. Other dirty
    # files in Verl are not automatically reset or cleaned.
    p = subprocess.run(['git', '-C', str(SFT.parents[2]), 'diff', '--', 'verl/trainer/sft_trainer.py'], capture_output=True, text=True, check=True)
    if p.stdout.strip():
        # This exact pre-existing version has been reviewed:
        # only configurable DistributedSampler shuffle was added.
        assert hashlib.sha256(data.encode()).hexdigest() == (
            '3c52e56b140a68eb9aac289b3b2f8694bb1a630dfae94507c02bc40028c9be14'
        ), 'UNKNOWN_PREEXISTING_SFT_DIFF'

        staged = subprocess.run(
            ['git', '-C', str(SFT.parents[2]), 'diff', '--cached',
             '--', 'verl/trainer/sft_trainer.py'],
            capture_output=True, text=True, check=True
        )
        assert not staged.stdout.strip(), 'UNREVIEWED_STAGED_DIFF'
        print('APPROVED_EXISTING_SAMPLER_DIFF_PRESERVED')
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup.write_text(data, encoding='utf-8')
    SFT.write_text(data.replace(OLD, NEW, 1), encoding='utf-8')
    compile(SFT.read_text(encoding='utf-8'), str(SFT), 'exec')
    print('PATCH_APPLIED; backup:', backup)
    subprocess.run(['git', '-C', str(SFT.parents[2]), 'diff', '--', 'verl/trainer/sft_trainer.py'], check=True)

if __name__ == '__main__':
    main()
