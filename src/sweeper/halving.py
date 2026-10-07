"""Scoring a segment's metric history, and the settings of opt-in successive halving."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

DIRECTIONS = ('min', 'max')
# the IQR prune skips a rung with fewer configs, their quartiles say too little
MIN_PRUNE_CONFIGS = 4


@dataclass(frozen=True)
class Score:
    """How a segment's metric history turns into the one number stored per repeat."""

    keys: tuple[str, ...]  # history keys, averaged with equal weight
    direction: Callable[[str], str]  # key -> 'min' or 'max', all keys must agree
    top_fraction: float  # share of each key's best epochs that gets averaged, in (0, 1]

    def __post_init__(self) -> None:
        # a bare str would silently score each of its letters as a key
        if isinstance(self.keys, str) or not self.keys:
            raise ValueError(f'keys must be a non-empty tuple of str, got {self.keys!r}')
        if not 0 < self.top_fraction <= 1:
            raise ValueError(f'top_fraction must be in (0, 1], got {self.top_fraction}')
        self.sense()

    def sense(self) -> str:
        """The direction all keys share, 'min' or 'max'.

        Returns:
            'min' when lower is better, 'max' when higher is.

        Raises:
            ValueError: if direction returns anything else for a key, or two keys disagree.
        """
        got = {}
        for key in self.keys:
            d = self.direction(key)
            if d not in DIRECTIONS:
                raise ValueError(f"direction({key!r}) must return 'min' or 'max', got {d!r}")
            got[key] = d

        if len(set(got.values())) > 1:
            raise ValueError(f'all score keys must share one direction, got {got}')
        return got[self.keys[0]]


def score_history(history: Mapping[str, Sequence[float]], score: Score) -> float:
    """Mean over keys of the mean of each key's best top_fraction of epochs.

    Args:
        history: key -> one value per epoch, as train_until returns it.
        score: keys, direction, and top_fraction to use.

    Returns:
        The score, in the keys' own direction. NaN epochs rank after every finite one, so the
        score is NaN only when a key has fewer than k finite values.

    Raises:
        ValueError: if a key is missing from history or its list is empty.
    """
    sign = -1 if score.sense() == 'max' else 1
    per_key = []
    for key in score.keys:
        values = history.get(key)
        if not values:
            raise ValueError(f'score key {key!r} missing or empty, history has {sorted(history)}')

        k = max(1, math.ceil(len(values) * score.top_fraction))
        # NaN compares false both ways, so a plain sort could rank it anywhere
        best = sorted(values, key=lambda v: (math.isnan(v), sign * v))[:k]
        per_key.append(sum(best) / k)
    return float(sum(per_key) / len(per_key))


@dataclass(frozen=True)
class Halving:
    """Successive halving settings, passing one to run_sweep is the only way to turn it on.

    The IQR prune runs only when outlier_fence is set, and it runs before the survival cut.
    So survival_fraction 1.0 with a fence gives an IQR-only prune.
    """

    rung_epochs: int  # epochs per rung, the last rung always ends at the sweep's epochs
    survival_fraction: float  # share of configs kept after each rung, rounded up, at least 1
    outlier_fence: float | None = None  # IQR fence of the prune before the cut, None skips it

    def __post_init__(self) -> None:
        if self.rung_epochs < 1:
            raise ValueError(f'rung_epochs must be >= 1, got {self.rung_epochs}')
        if not 0 < self.survival_fraction <= 1:
            raise ValueError(f'survival_fraction must be in (0, 1], got {self.survival_fraction}')
        if self.outlier_fence is not None and self.outlier_fence <= 0:
            raise ValueError(f'outlier_fence must be None or > 0, got {self.outlier_fence}')


def rung_targets(epochs: int, halving: Halving | None) -> list[int]:
    """Epoch targets of the rungs, every rung_epochs and always ending at epochs.

    Args:
        epochs: epochs per repeat.
        halving: halving settings, None gives the one rung of a flat run.

    Returns:
        Ascending targets, rung_epochs 2 and epochs 5 give [2, 4, 5].
    """
    if halving is None:
        return [epochs]

    targets = list(range(halving.rung_epochs, epochs + 1, halving.rung_epochs))
    if not targets or targets[-1] != epochs:
        targets.append(epochs)
    return targets


def outliers(rung_scores: Mapping[str, float], fence: float, direction: str) -> list[str]:
    """Configs whose score sits past the IQR fence on the bad side, and every NaN score.

    Args:
        rung_scores: config name -> its rung score.
        fence: Tukey multiplier of the IQR, bigger cuts less.
        direction: 'min' cuts above Q3 + fence * IQR, 'max' cuts below Q1 - fence * IQR.

    Returns:
        Names to cut. NaN scores are cut whenever a finite one exists, the fence only runs with
        at least MIN_PRUNE_CONFIGS finite scores.

    Raises:
        ValueError: if direction is not 'min' or 'max'.
    """
    _check_direction(direction)
    finite = {name: s for name, s in rung_scores.items() if not math.isnan(s)}
    # with no finite score left the survival cut decides, so the fence never empties a rung
    nan = [name for name in rung_scores if name not in finite] if finite else []
    if len(finite) < MIN_PRUNE_CONFIGS:
        return nan

    vals = sorted(finite.values())
    n = len(vals)
    # quartiles are plain picks from the sorted scores, no interpolation
    q1 = vals[n // 4]
    q3 = vals[3 * n // 4]
    if direction == 'min':
        bad = q3 + fence * (q3 - q1)
        return nan + [name for name, s in finite.items() if s > bad]
    bad = q1 - fence * (q3 - q1)
    return nan + [name for name, s in finite.items() if s < bad]


def survivors(rung_scores: Mapping[str, float], fraction: float, direction: str) -> list[str]:
    """The ceil(n * fraction) best configs, at least 1, best first.

    Args:
        rung_scores: config name -> its rung score.
        fraction: share to keep, in (0, 1].
        direction: 'min' when lower scores are better, 'max' when higher are.

    Returns:
        Kept names, ties broken by name. NaN scores rank after every finite one.

    Raises:
        ValueError: if fraction is outside (0, 1] or direction is not 'min' or 'max'.
    """
    _check_direction(direction)
    if not 0 < fraction <= 1:
        raise ValueError(f'fraction must be in (0, 1], got {fraction}')

    n_keep = max(1, math.ceil(len(rung_scores) * fraction))
    sign = 1 if direction == 'min' else -1

    def rank(name: str) -> tuple[bool, float, str]:
        s = rung_scores[name]
        # NaN goes last with a 0 stand-in, so only the name orders NaN configs among themselves
        if math.isnan(s):
            return (True, 0.0, name)
        return (False, sign * s, name)

    # the name tie-break keeps the cut independent of the configs' order
    ranked = sorted(rung_scores, key=rank)
    return ranked[:n_keep]


def _check_direction(direction: str) -> None:
    """Raise ValueError unless direction is 'min' or 'max'."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
