#!/usr/bin/env python3
from __future__ import annotations
import json, os
from pathlib import Path
import torch
from offline_prefix_support_verl_replay_manager_formal_v1 import OfflinePrefixSupportFormalReplayManager

def atomic_json(path: Path, obj):
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

class OfflinePrefixSupportAllOnesReplayManagerV2(OfflinePrefixSupportFormalReplayManager):
    # Engineering-only arm: same S replay, but explicit weight=1 on valid response
    # tokens and weight=0 on padding.
    def generate_sequences(self, prompts):
        out=super().generate_sequences(prompts)
        rm=out.batch["response_mask"]
        w=rm.to(dtype=torch.float32).detach().clone()
        w.requires_grad_(False)
        assert w.shape==rm.shape and not w.requires_grad
        assert torch.equal(w>0,rm.bool())
        assert torch.all(w[~rm.bool()]==0)
        out.batch["distillation_token_weights"]=w
        arm_run=Path(str(self.config.trainer.default_local_dir)).parent
        atomic_json(arm_run/"token_weight_transport.json",{
          "status":"PASS_ALL_ONES_WEIGHT_TRANSPORT",
          "selected_tokens":int(w.sum().item()),
          "valid_response_tokens":int(rm.sum().item()),
          "padding_weight_nonzero":int((w[~rm.bool()]!=0).sum().item()),
          "requires_grad":bool(w.requires_grad),
        })
        print(f"DISTILLATION_TOKEN_WEIGHTS_TRANSPORT_PASS selected={int(w.sum())} valid={int(rm.sum())}",flush=True)
        return out
