#!/usr/bin/env python3

from __future__ import annotations

import torch


DEFAULT_MASS_TOLERANCE = 1.0e-5


def _require_finite(
    name: str,
    x: torch.Tensor,
) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(
            f"{name} contains non-finite values"
        )


def _project_one_topk_other(
    topk_log_probs: torch.Tensor,
    *,
    mass_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Independent reference projection:

        explicit categories = supplied Teacher-TopK coordinates
        OTHER = residual probability mass

    The reference intentionally computes in float64.

    A tiny mass > 1 is tolerated only as floating-point roundoff.
    Such a case gets residual=0 followed by K+1 renormalization.

    A material mass > 1 is a hard failure.
    """

    lp = topk_log_probs.to(torch.float64)

    _require_finite(
        "topk_log_probs",
        lp,
    )

    probs = lp.exp()

    _require_finite(
        "topk_probs",
        probs,
    )

    mass = probs.sum(
        dim=-1,
    )

    _require_finite(
        "explicit_mass",
        mass,
    )

    excessive = (
        mass
        > 1.0 + mass_tolerance
    )

    if bool(excessive.any()):
        worst = float(
            mass[excessive].max()
        )

        raise ValueError(
            "explicit probability mass exceeds "
            f"1+tolerance: worst={worst} "
            f"tolerance={mass_tolerance}"
        )

    residual = (
        1.0 - mass
    ).clamp_min(0.0)

    projected = torch.cat(
        [
            probs,
            residual.unsqueeze(-1),
        ],
        dim=-1,
    )

    z = projected.sum(
        dim=-1,
        keepdim=True,
    )

    if bool((z <= 0).any()):
        raise ValueError(
            "projected distribution has "
            "non-positive normalization"
        )

    projected = projected / z

    _require_finite(
        "projected_distribution",
        projected,
    )

    return projected, mass


def project_teacher_topk_other(
    teacher_topk_log_probs: torch.Tensor,
    student_on_teacher_topk_log_probs: torch.Tensor,
    *,
    mass_tolerance: float = DEFAULT_MASS_TOLERANCE,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build the common Teacher-TopK + OTHER projected space.

    Returns:
        teacher_projected
        student_projected
        raw_teacher_explicit_mass
        raw_student_explicit_mass
    """

    if (
        teacher_topk_log_probs.shape
        != student_on_teacher_topk_log_probs.shape
    ):
        raise ValueError(
            "Teacher/Student top-k logprob "
            "shape mismatch: "
            f"{teacher_topk_log_probs.shape} vs "
            f"{student_on_teacher_topk_log_probs.shape}"
        )

    teacher_projected, teacher_mass = (
        _project_one_topk_other(
            teacher_topk_log_probs,
            mass_tolerance=mass_tolerance,
        )
    )

    student_projected, student_mass = (
        _project_one_topk_other(
            student_on_teacher_topk_log_probs,
            mass_tolerance=mass_tolerance,
        )
    )

    return (
        teacher_projected,
        student_projected,
        teacher_mass,
        student_mass,
    )


def projected_fkl(
    teacher_projected: torch.Tensor,
    student_projected: torch.Tensor,
) -> torch.Tensor:
    """
    KL(P_T' || P_S').

    No final clamp is performed.
    """

    p = teacher_projected.to(
        torch.float64
    )

    q = student_projected.to(
        torch.float64
    )

    if p.shape != q.shape:
        raise ValueError(
            f"shape mismatch: {p.shape} vs {q.shape}"
        )

    _require_finite("teacher_projected", p)
    _require_finite("student_projected", q)

    if bool((p < 0).any()):
        raise ValueError(
            "negative Teacher probability"
        )

    if bool((q < 0).any()):
        raise ValueError(
            "negative Student probability"
        )

    impossible = (
        (p > 0)
        & (q <= 0)
    )

    if bool(impossible.any()):
        raise FloatingPointError(
            "FKL has positive Teacher mass "
            "where Student mass is zero"
        )

    # torch.special.xlogy(0, 0) = 0,
    # which gives the correct zero-mass convention.
    out = (
        torch.special.xlogy(p, p)
        - torch.special.xlogy(p, q)
    ).sum(dim=-1)

    _require_finite(
        "projected_fkl",
        out,
    )

    return out


def projected_jsd(
    teacher_projected: torch.Tensor,
    student_projected: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Generalized JSD exactly in the GKD convention:

        M = beta * P_T' + (1-beta) * P_S'

        JSD_beta =
            beta * KL(P_T' || M)
            + (1-beta) * KL(P_S' || M)

    No division by beta or (1-beta).
    No final clamp.
    """

    if not (
        0.0 < beta < 1.0
    ):
        raise ValueError(
            f"beta must be in (0,1), got {beta}"
        )

    p = teacher_projected.to(
        torch.float64
    )

    q = student_projected.to(
        torch.float64
    )

    if p.shape != q.shape:
        raise ValueError(
            f"shape mismatch: {p.shape} vs {q.shape}"
        )

    _require_finite("teacher_projected", p)
    _require_finite("student_projected", q)

    if bool((p < 0).any()):
        raise ValueError(
            "negative Teacher probability"
        )

    if bool((q < 0).any()):
        raise ValueError(
            "negative Student probability"
        )

    m = (
        beta * p
        + (1.0 - beta) * q
    )

    _require_finite(
        "mixture",
        m,
    )

    kl_pm = (
        torch.special.xlogy(p, p)
        - torch.special.xlogy(p, m)
    ).sum(dim=-1)

    kl_qm = (
        torch.special.xlogy(q, q)
        - torch.special.xlogy(q, m)
    ).sum(dim=-1)

    out = (
        beta * kl_pm
        + (1.0 - beta) * kl_qm
    )

    _require_finite(
        "projected_jsd",
        out,
    )

    return out
