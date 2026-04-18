from typing import Mapping, TypeAlias, TypeVar

K = TypeVar("K")

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

ActionProbLike: TypeAlias = Mapping[ActType, float]

def argmax(mapping: Mapping[K, float]) -> K:
    """Return a key from `mapping` with the largest value.

    Ties are broken by first-seen iteration order.
    Raises ValueError if `mapping` is empty.
    """
    max_key: K | None = None
    max_val = float("-inf")
    for k, v in mapping.items():
        if v > max_val:
            max_val = v
            max_key = k
    if max_key is None:
        raise ValueError("argmax() received an empty mapping")
    return max_key

def soften_policy(policy: Mapping[ObsType, ActionProbLike[ActType]], epsilon: float) -> dict[ObsType, dict[ActType, float]]:
    """Return a new policy that is an epsilon-softened version of `policy`."""
    softened = {}
    for state, action_probs in policy.items():
        num_actions = len(action_probs)
        soft_prob = epsilon / num_actions
        softened[state] = {a: p * (1 - epsilon) + soft_prob for a, p in action_probs.items()}
    return softened

def parse_human_int(text: str) -> int:
    value = text.strip().replace(",", "")
    suffix = value[-1].lower()
    if suffix == "k":
        return int(float(value[:-1]) * 1_000)
    if suffix == "m":
        return int(float(value[:-1]) * 1_000_000)
    if suffix == "b":
        return int(float(value[:-1]) * 1_000_000_000)
    return int(value)
