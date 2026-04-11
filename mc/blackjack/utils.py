from typing import Mapping, TypeVar

K = TypeVar("K")
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
