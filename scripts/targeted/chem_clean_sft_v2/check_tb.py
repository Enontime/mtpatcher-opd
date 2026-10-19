#!/usr/bin/env python3
"""Checks real on-disk TensorBoard event scalars; never synthesizes events."""
import argparse
import math
from collections import defaultdict
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ('train/loss','train/lr','train/grad_norm')

def inspect(root, expected):
    p = Path(root).resolve()
    ev = sorted(p.rglob('events.out.tfevents.*'))
    assert len(ev) == 1, f'EXPECTED_ONE_NATIVE_EVENT_WRITER got={len(ev)} files={ev[:6]}'
    assert str(ev[0].parent.resolve()) == str(p), 'UNEXPECTED_TB_SUBDIRECTORY'
    acc = EventAccumulator(str(p), size_guidance={'scalars':0,'tensors':0})
    acc.Reload()
    scalar_tags = set(acc.Tags().get('scalars', []))
    tensor_tags = set(acc.Tags().get('tensors', []))
    for tag in TAGS:
        if tag in scalar_tags:
            events = [(int(x.step), float(x.value)) for x in acc.Scalars(tag)]
        elif tag in tensor_tags:
            from tensorboard.util.tensor_util import make_ndarray
            events = [(int(x.step), float(make_ndarray(x.tensor_proto).reshape(-1)[0])) for x in acc.Tensors(tag)]
        else:
            raise RuntimeError(f'MISSING_REAL_TENSORBOARD_TAG {tag} scalar={scalar_tags} tensor={tensor_tags}')
        steps = [a for a,b in events]
        assert steps == expected, f'{tag} unexpected steps={steps[:15]} expected={expected[:15]}'
        assert all(math.isfinite(v) for _,v in events), f'{tag} HAS_NAN_INF'
        print(f'{tag} count={len(events)} min={min(x[1] for x in events):.7g} max={max(x[1] for x in events):.7g}')
    print(f'REAL_TENSORBOARD_PASS {p} steps={expected[0]}..{expected[-1]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tb_dir')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--last', type=int, required=True)
    args = parser.parse_args()
    inspect(args.tb_dir, list(range(args.start, args.last+1)))
