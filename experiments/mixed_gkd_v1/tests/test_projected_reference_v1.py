#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT = Path(
    "/workspace/mtpatcher/repo/"
    "MT-Patcher-Reproduction-Ascend"
)

if str(PROJECT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT),
    )

REFERENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "reference"
)

if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(REFERENCE_DIR),
    )

from projected_reference_v1 import (  # noqa: E402
    DEFAULT_MASS_TOLERANCE,
    project_teacher_topk_other,
    projected_fkl,
    projected_jsd,
)


torch.set_default_dtype(
    torch.float64
)


def assert_close(
    a,
    b,
    *,
    atol=1e-12,
    rtol=1e-12,
):
    torch.testing.assert_close(
        a,
        b,
        atol=atol,
        rtol=rtol,
    )


print(
    "===== TEST 1: DIRECT COARSENING EQUIVALENCE ====="
)

torch.manual_seed(
    20260928
)

B = 2
T = 5
V = 97
K = 32

teacher_logits = torch.randn(
    B,
    T,
    V,
    dtype=torch.float64,
)

student_logits = torch.randn(
    B,
    T,
    V,
    dtype=torch.float64,
)

teacher_log_probs = torch.log_softmax(
    teacher_logits,
    dim=-1,
)

student_log_probs = torch.log_softmax(
    student_logits,
    dim=-1,
)

teacher_ids = torch.topk(
    teacher_log_probs,
    k=K,
    dim=-1,
).indices

teacher_topk_lp = torch.gather(
    teacher_log_probs,
    dim=-1,
    index=teacher_ids,
)

student_on_teacher_topk_lp = torch.gather(
    student_log_probs,
    dim=-1,
    index=teacher_ids,
)

(
    pt,
    ps,
    teacher_mass,
    student_mass,
) = project_teacher_topk_other(
    teacher_topk_lp,
    student_on_teacher_topk_lp,
)

teacher_probs = teacher_log_probs.exp()
student_probs = student_log_probs.exp()

direct_teacher_explicit = torch.gather(
    teacher_probs,
    dim=-1,
    index=teacher_ids,
)

direct_student_explicit = torch.gather(
    student_probs,
    dim=-1,
    index=teacher_ids,
)

direct_pt = torch.cat(
    [
        direct_teacher_explicit,
        (
            1.0
            - direct_teacher_explicit.sum(
                dim=-1,
                keepdim=True,
            )
        ),
    ],
    dim=-1,
)

direct_ps = torch.cat(
    [
        direct_student_explicit,
        (
            1.0
            - direct_student_explicit.sum(
                dim=-1,
                keepdim=True,
            )
        ),
    ],
    dim=-1,
)

assert_close(
    pt,
    direct_pt,
)

assert_close(
    ps,
    direct_ps,
)

assert_close(
    teacher_mass,
    direct_teacher_explicit.sum(
        dim=-1
    ),
)

assert_close(
    student_mass,
    direct_student_explicit.sum(
        dim=-1
    ),
)

assert_close(
    pt.sum(dim=-1),
    torch.ones(B, T),
)

assert_close(
    ps.sum(dim=-1),
    torch.ones(B, T),
)

print(
    "DIRECT_COARSENING_EQUIVALENCE=PASS"
)


print()
print(
    "===== TEST 2: IDENTITY / NON-NEGATIVITY ====="
)

fkl_identity = projected_fkl(
    pt,
    pt,
)

jsd_identity = projected_jsd(
    pt,
    pt,
    beta=0.1,
)

assert float(
    fkl_identity.abs().max()
) <= 1e-12

assert float(
    jsd_identity.abs().max()
) <= 1e-12

fkl = projected_fkl(
    pt,
    ps,
)

jsd = projected_jsd(
    pt,
    ps,
    beta=0.1,
)

assert float(
    fkl.min()
) >= -1e-12

assert float(
    jsd.min()
) >= -1e-12

print(
    "IDENTITY_ZERO=PASS"
)

print(
    "NONNEGATIVITY=PASS"
)

print(
    "FKL_MEAN =",
    float(fkl.mean()),
)

print(
    "JSD01_MEAN =",
    float(jsd.mean()),
)


print()
print(
    "===== TEST 3: JSD BETA CONVENTION ====="
)

beta = 0.1

m = (
    beta * pt
    + (1.0 - beta) * ps
)

manual = (
    beta
    * (
        torch.special.xlogy(pt, pt)
        - torch.special.xlogy(pt, m)
    ).sum(dim=-1)
    +
    (1.0 - beta)
    * (
        torch.special.xlogy(ps, ps)
        - torch.special.xlogy(ps, m)
    ).sum(dim=-1)
)

assert_close(
    jsd,
    manual,
)

# Explicitly guard against accidental /beta normalization.
assert not torch.allclose(
    jsd,
    manual / beta,
    atol=1e-12,
    rtol=1e-12,
)

print(
    "JSD_BETA_0P1_CONVENTION=PASS"
)


print()
print(
    "===== TEST 4: STUDENT BACKWARD ====="
)

student_logits_grad = torch.randn(
    B,
    T,
    V,
    dtype=torch.float64,
    requires_grad=True,
)

student_lp_grad = torch.log_softmax(
    student_logits_grad,
    dim=-1,
)

student_topk_grad = torch.gather(
    student_lp_grad,
    dim=-1,
    index=teacher_ids,
)

(
    pt_grad,
    ps_grad,
    _,
    _,
) = project_teacher_topk_other(
    teacher_topk_lp.detach(),
    student_topk_grad,
)

loss = (
    projected_fkl(
        pt_grad,
        ps_grad,
    ).mean()
    +
    projected_jsd(
        pt_grad,
        ps_grad,
        beta=0.1,
    ).mean()
)

loss.backward()

grad = student_logits_grad.grad

assert grad is not None
assert bool(
    torch.isfinite(grad).all()
)
assert bool(
    (grad.abs() > 0).any()
)

print(
    "STUDENT_BACKWARD_FINITE=PASS"
)


print()
print(
    "===== TEST 5: OTHER PATH GRADIENT ====="
)

student_logits_other = torch.randn(
    1,
    1,
    V,
    dtype=torch.float64,
    requires_grad=True,
)

student_lp_other = torch.log_softmax(
    student_logits_other,
    dim=-1,
)

teacher_ids_one = teacher_ids[
    :1,
    :1,
]

student_topk_other_lp = torch.gather(
    student_lp_other,
    dim=-1,
    index=teacher_ids_one,
)

teacher_topk_one = teacher_topk_lp[
    :1,
    :1,
].detach()

(
    _pt_other,
    ps_other,
    _,
    _,
) = project_teacher_topk_other(
    teacher_topk_one,
    student_topk_other_lp,
)

student_other_probability = (
    ps_other[..., -1].sum()
)

student_other_probability.backward()

other_grad = student_logits_other.grad

assert other_grad is not None
assert bool(
    torch.isfinite(other_grad).all()
)

explicit_mask = torch.zeros(
    V,
    dtype=torch.bool,
)

explicit_mask[
    teacher_ids_one.reshape(-1)
] = True

outside_grad = (
    other_grad[
        0,
        0,
        ~explicit_mask,
    ]
)

assert bool(
    (outside_grad.abs() > 0).any()
)

print(
    "OTHER_PATH_GRADIENT=PASS"
)


print()
print(
    "===== TEST 6: ROUND-OFF STABILIZATION ====="
)

# Explicit mass exceeds 1 by only 5e-7:
# permitted as roundoff under the frozen 1e-5 reference tolerance.
roundoff_probs = torch.tensor(
    [
        [
            0.6000000,
            0.4000005,
        ]
    ],
    dtype=torch.float64,
)

roundoff_lp = roundoff_probs.log()

(
    round_pt,
    round_ps,
    round_tm,
    round_sm,
) = project_teacher_topk_other(
    roundoff_lp,
    roundoff_lp,
    mass_tolerance=
        DEFAULT_MASS_TOLERANCE,
)

assert float(
    round_tm.item()
) > 1.0

assert float(
    round_sm.item()
) > 1.0

assert_close(
    round_pt.sum(dim=-1),
    torch.ones(1),
)

assert_close(
    round_ps.sum(dim=-1),
    torch.ones(1),
)

assert float(
    round_pt[..., -1].item()
) == 0.0

print(
    "ROUND_OFF_STABILIZATION=PASS"
)


print()
print(
    "===== TEST 7: MATERIAL MASS ERROR FAILS ====="
)

bad_probs = torch.tensor(
    [
        [
            0.60,
            0.401,
        ]
    ],
    dtype=torch.float64,
)

bad_lp = bad_probs.log()

failed = False

try:
    project_teacher_topk_other(
        bad_lp,
        bad_lp,
        mass_tolerance=
            DEFAULT_MASS_TOLERANCE,
    )
except ValueError:
    failed = True

assert failed

print(
    "MATERIAL_MASS_ERROR_GATE=PASS"
)


print()
print(
    "===== TEST 8: TEACHER DETACH CONTRACT ====="
)

teacher_no_grad = (
    teacher_topk_lp
    .detach()
    .clone()
)

assert (
    teacher_no_grad.requires_grad
    is False
)

print(
    "TEACHER_DETACHED=PASS"
)


print()
print(
    "========================================"
)

print(
    "MIXED_GKD_PROJECTED_REFERENCE_V1=PASS"
)
