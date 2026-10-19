"""Local CPU tests: validate mask extraction and source-contract invariants.
Not a native-Verl integration test and NOT a scientific numerical parity test.
"""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def stub(path, **items):
    m=types.ModuleType(path)
    m.__dict__.update(items)
    sys.modules[path]=m
    return m


def load_manager():
    for k in ['verl','verl.experimental','verl.experimental.agent_loop',
              'verl.experimental.teacher_loop','verl.experimental.teacher_loop.teacher_manager',
              'verl.utils','verl.utils.ray_utils','verl.utils.config']:
        stub(k)
    stub('verl.experimental.agent_loop',AgentLoopManager=type('AgentLoopManager',(object,),{}))
    stub('verl.experimental.teacher_loop.teacher_manager',
         AsyncTeacherLLMServerManager=object,_pad_teacher_outputs=lambda *a,**kw:None)
    stub('verl.utils.config',omega_conf_to_dataclass=lambda x:x)
    stub('verl.utils.ray_utils',auto_await=lambda fn:fn)
    spec=importlib.util.spec_from_file_location('chemistry_onpolicy_manager_v1',
              ROOT/'scripts/opd/chemistry_onpolicy_manager_v1.py')
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class Output:
    def __init__(self,mask,attention):
        self.batch={
            'responses':torch.tensor([[17,18,19,0]],dtype=torch.long),
            'response_mask':torch.tensor([mask],dtype=torch.long),
            'attention_mask':torch.tensor([attention],dtype=torch.long),
        }


def main():
    m=load_manager()
    assert m.DIRECT_PROMPT.startswith('Translate the following text')
    assert m.extract_response(Output([1,1,1,0],[1,1,1,1,1,1,0]),0)==[17,18,19]
    for mask,attention in [([1,0,1,0],[1,1,1,1,1,1,0]),
                           ([1,1,0,0],[1,1,1,1,1,1,0]),
                           ([0,0,0,0],[1,1,1,0,0,0,0])]:
        try:
            m.extract_response(Output(mask,attention),0)
        except RuntimeError as exc:
            assert 'CONTRACT_FAIL' in str(exc)
        else:
            raise AssertionError((mask,attention))
    print('LOCAL_MASK_AND_SINGLE_TURN_GATES=PASS')


if __name__=='__main__':
    main()
