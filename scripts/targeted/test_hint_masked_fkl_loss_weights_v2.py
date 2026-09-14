#!/usr/bin/env python3
from __future__ import annotations
import json
import torch

# Padding deliberately carries huge values. DENSE and ALL-ONES are NOT required
# to be raw-tensor equal on padding; native agg_loss ignores those positions.
response_mask=torch.tensor([[1,1,1,0],[1,1,0,0]],dtype=torch.bool)
losses=torch.tensor(
    [[0.2,0.7,1.3,99.0],[0.5,2.0,88.0,77.0]],
    dtype=torch.float32,
    requires_grad=True,
)

def apply_weights(loss_mat,mask,weights):
    if weights is None:
        return loss_mat
    assert weights.shape==loss_mat.shape==mask.shape
    assert not weights.requires_grad
    assert torch.isfinite(weights).all()
    assert (weights>=0).all()
    assert (weights[~mask]==0).all()
    return loss_mat*weights.detach().to(loss_mat.dtype)

def full_response_token_mean(loss_mat,mask):
    m=mask.to(loss_mat.dtype)
    return (loss_mat*m).sum()/m.sum()

dense=apply_weights(losses,response_mask,None)
ones=apply_weights(losses,response_mask,response_mask.float().detach())

# What must be equal is the VALID response-token loss and the actual objective.
assert torch.equal(dense[response_mask],ones[response_mask])
assert torch.equal(
    full_response_token_mean(dense,response_mask),
    full_response_token_mean(ones,response_mask),
)

g_dense=torch.autograd.grad(
    full_response_token_mean(dense,response_mask),
    losses,
    retain_graph=True,
)[0]
g_ones=torch.autograd.grad(
    full_response_token_mean(ones,response_mask),
    losses,
    retain_graph=True,
)[0]
assert torch.equal(g_dense,g_ones)

onehot=torch.zeros_like(losses).detach()
onehot[0,1]=1.0
got=full_response_token_mean(
    apply_weights(losses,response_mask,onehot),
    response_mask,
)
manual=losses[0,1]/response_mask.sum()
assert torch.allclose(got,manual,atol=0,rtol=0)

bad_padding=response_mask.float().detach().clone()
bad_padding[0,3]=1.0
try:
    apply_weights(losses,response_mask,bad_padding)
    raise RuntimeError("padding gate failed")
except AssertionError:
    pass

grad_weights=response_mask.float().clone().requires_grad_(True)
try:
    apply_weights(losses,response_mask,grad_weights)
    raise RuntimeError("detached gate failed")
except AssertionError:
    pass

print(json.dumps({
 "status":"PASS_HINT_MASKED_FKL_LOSS_UNIT_GATES",
 "weights_None_equals_all_ones_on_valid_tokens":True,
 "weights_None_equals_all_ones_objective":True,
 "all_ones_gradient_equal":True,
 "one_hot_matches_manual_full_denominator_formula":True,
 "padding_nonzero_rejected":True,
 "requires_grad_weights_rejected":True,
 "note":"raw padding entries may differ and are intentionally excluded by response_mask",
},indent=2,sort_keys=True))
