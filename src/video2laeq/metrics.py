"""Υπολογισμός LAeq, LAF10, LAF90 από δείγματα 1/sec και έλεγχος αληθοφάνειας."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    t: float
    value: float | None
    valid: bool = True
    reason: str = ""


@dataclass(frozen=True)
class Result:
    laeq: float
    laf10: float
    laf90: float
    lmin: float
    lmax: float
    n_valid: int
    n_total: int
    duration: float


def validate(samples: list[Sample], lo: float = 20.0, hi: float = 140.0,
             max_jump: float = 15.0) -> list[Sample]:
    """Σημειώνει ως άκυρα: None, εκτός ορίων, ή απομονωμένα άλματα > max_jump."""
    for s in samples:
        if s.value is None:
            s.valid = False
            s.reason = s.reason or "μη αναγνώσιμο"
        elif not lo <= s.value <= hi:
            s.valid = False
            s.reason = f"εκτός ορίων ({s.value:.1f})"
    valid_idx = [i for i, s in enumerate(samples) if s.valid]
    prev_v: float | None = None
    for k, i in enumerate(valid_idx):
        v = samples[i].value
        next_v = samples[valid_idx[k + 1]].value if k + 1 < len(valid_idx) else None
        neighbours = [n for n in (prev_v, next_v) if n is not None]
        if neighbours and all(abs(v - n) > max_jump for n in neighbours):
            samples[i].valid = False
            samples[i].reason = f"απομονωμένο άλμα ({v:.1f})"
        else:
            prev_v = v
    return samples


def aggregate(values: list[float], tolerance: float = 1.0) -> float | None:
    """Ένα δείγμα από τις αναγνώσεις των καρέ ενός δευτερολέπτου.

    Με μία ανάγνωση την επιστρέφει. Με περισσότερες, κρατά τη μεγαλύτερη ομάδα τιμών που
    απέχουν ≤ tolerance dB μεταξύ τους και επιστρέφει τη διάμεσό της· αν καμία δύο δεν
    συμφωνούν, η ένδειξη θεωρείται αναξιόπιστη (None).
    """
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    best: list[float] = []
    for i, v in enumerate(vals):
        group = [u for u in vals[i:] if u - v <= tolerance]
        if len(group) > len(best):
            best = group
    if len(best) < 2:
        return None
    return float(np.median(best))


def laeq(levels: np.ndarray) -> float:
    return float(10.0 * math.log10(np.mean(10.0 ** (levels / 10.0))))


def compute(samples: list[Sample], duration: float | None = None) -> Result:
    levels = np.array([s.value for s in samples if s.valid], dtype=float)
    if levels.size == 0:
        raise ValueError("Δεν υπάρχουν έγκυρα δείγματα")
    if duration is None:
        duration = (samples[-1].t + 1.0) if samples else 0.0
    return Result(
        laeq=laeq(levels),
        laf10=float(np.percentile(levels, 90)),
        laf90=float(np.percentile(levels, 10)),
        lmin=float(levels.min()),
        lmax=float(levels.max()),
        n_valid=int(levels.size),
        n_total=len(samples),
        duration=float(duration),
    )
