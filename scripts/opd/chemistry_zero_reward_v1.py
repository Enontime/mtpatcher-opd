"""Neutral reward transport for frozen Chemistry on-policy OPD.

The native AgentLoop asks for this score even with task rewards and policy
gradient disabled. This score must NEVER be interpreted as translation quality.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "chemistry":
        raise ValueError(f"chemistry_zero_reward_v1: wrong data_source={data_source!r}")
    if ground_truth != "UNUSED_ZERO_REWARD":
        raise ValueError(f"chemistry_zero_reward_v1: unexpected ground_truth={ground_truth!r}")
    return 0.0
