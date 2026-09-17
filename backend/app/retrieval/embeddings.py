"""Small local term-vector embeddings used to keep the prototype self-contained."""
from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text: str) -> dict[str, float]:
    counts = Counter(tokenize(text))
    length = math.sqrt(sum(count * count for count in counts.values()))
    return {term: count / length for term, count in counts.items()} if length else {}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())
