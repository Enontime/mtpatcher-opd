"""Control-plane-only reward adapter for pure direct-distillation OPD.

Verl's AgentLoop requires a reward_score to materialize trajectories even when
task rewards and policy-gradient training are disabled.  This callback supplies
that required plumbing value.

It MUST NOT be interpreted as an MT quality reward.
"""


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    if data_source != "default":
        raise ValueError(
            f"constant_zero_reward is scoped to data_source='default', got {data_source!r}"
        )

    return 0.0
