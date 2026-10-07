"""Crash-safe runner for a list of configs times repeats, resumable by config name.

A segment trains one (config, repeat) up to a rung's target epoch. It commits in two steps,
the score into the state file first and then the rung checkpoint, so a kill anywhere resumes
from the last committed step. Layout under out_dir:
    sweep_state.json
    sweep.lock, held by the one run_sweep that writes here
    <config>/repeat<r>/latest.pth, written by the Trial while it trains
    <config>/repeat<r>/rung<i>.pth, committed, one per rung
"""

import dataclasses
import fcntl
import json
import logging
import math
import os
import re
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sweeper.config import Config
from sweeper.halving import Halving, Score, outliers, rung_targets, score_history, survivors
from sweeper.runlog import RunLogger
from sweeper.seeds import config_seed, seed_everything
from sweeper.state import STATE_FILE, STATE_VERSION, load_state, write_json_atomic
from sweeper.trial import Trial

LATEST_CKPT = 'latest.pth'
LOCK_FILE = 'sweep.lock'
# no leading dot, so '.' and '..' can never be a config dir
_NAME_RE = re.compile(r'[A-Za-z0-9_-][A-Za-z0-9._-]*')
_MISSING = object()  # stands in for a params key only one side has

_log = logging.getLogger('sweeper')


def run_sweep(
    configs: Sequence[Config],
    trial_factory: Callable[[], Trial],
    out_dir: Path,
    *,
    sweep: str,
    epochs: int,
    base_seed: int,
    n_repeats: int = 1,
    resume: bool = False,
    logger: RunLogger | None = None,
    score: Score | None = None,
    halving: Halving | None = None,
) -> dict[str, Any]:
    """Train every config n_repeats times up to epochs, committing each segment to disk.

    Every config runs repeat 0 before any config runs repeat 1. A rerun with resume skips what
    is committed and picks the rest back up from its latest checkpoint.

    Args:
        configs: configs to run, names unique and filesystem safe.
        trial_factory: called once per segment, returns a fresh Trial.
        out_dir: holds the state file and one dir per config, created if missing.
        sweep: sweep name, mixed into derived seeds and checked on resume.
        epochs: epochs per repeat.
        base_seed: base of the derived seeds, a config's own seeds win over it.
        n_repeats: repeats per config.
        resume: continue the state file in out_dir instead of starting a new one.
        logger: one run per (config, repeat), None logs nothing.
        score: turns a segment's history into its stored score, None stores null.
        halving: successive halving settings, None runs every config to epochs.

    Returns:
        The final state, as written to out_dir/sweep_state.json.

    Raises:
        TypeError: if configs holds something other than Config.
        ValueError: on a bad config list, epochs, or n_repeats, on a state file that exists
            without resume or is missing with resume, on checkpoints left in a config's run dir
            without resume, and on a resume whose sweep, epochs, n_repeats, halving, score
            settings, configs, params, or seeds differ from the state.
        RuntimeError: if another run_sweep holds the out_dir lock, or a Trial's epoch after
            restore or train_until breaks the contract.
    """
    _check_args(configs, epochs, n_repeats)
    if halving is not None and score is None:
        raise ValueError('halving ranks configs by score, so it needs a Score, got score=None')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / STATE_FILE
    scoring = None
    if score is not None:
        # list, since a resume compares it with what came back from the JSON state
        scoring = {
            'keys': list(score.keys),
            'top_fraction': score.top_fraction,
            'sense': score.sense(),
        }
    head = {
        'sweep': sweep,
        'epochs': epochs,
        'n_repeats': n_repeats,
        'halving': None if halving is None else dataclasses.asdict(halving),
        'score': scoring,
    }
    targets = rung_targets(epochs, halving)

    lock_path = out_dir / LOCK_FILE
    # flock dies with the fd, so a killed sweep never leaves a stale lock, only the empty file
    with open(lock_path, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f'{lock_path} is held, another run_sweep writes to {out_dir}'
            ) from None

        state = _open_state(state_path, configs, head, targets, base_seed, resume)
        # rung runs past the last target once an early stop ends the sweep
        for i in range(state['rung'], len(targets)):
            live = [c for c in configs if state['configs'][c.name]['pruned_at'] is None]
            for r in range(n_repeats):
                for cfg in live:
                    run_dir = out_dir / cfg.name / f'repeat{r}'
                    _run_segment(
                        state, state_path, cfg, r, i, run_dir, trial_factory, logger, score
                    )

            if halving is None or i == len(targets) - 1:
                continue
            _prune(state, state_path, i, halving, score.sense())
            if state['rung'] == len(targets):
                break
    return state


def _check_args(configs: Sequence[Config], epochs: int, n_repeats: int) -> None:
    """Raise on a bad epochs or n_repeats, or a config list the state can't be keyed by."""
    if epochs < 1:
        raise ValueError(f'epochs must be >= 1, got {epochs}')
    if n_repeats < 1:
        raise ValueError(f'n_repeats must be >= 1, got {n_repeats}')
    if not configs:
        raise ValueError(f'configs is empty, got {configs!r}')

    seen = set()
    for c in configs:
        if not isinstance(c, Config):
            raise TypeError(f'configs must hold Config objects, got {type(c).__name__}')
        if not _NAME_RE.fullmatch(c.name):
            raise ValueError(
                f'config name {c.name!r} is not filesystem safe, use letters, digits, ".", "_", '
                'and "-", and no leading "."'
            )
        if c.name in seen:
            raise ValueError(f'duplicate config name {c.name!r}')
        seen.add(c.name)
        if c.seeds is not None and len(c.seeds) != n_repeats:
            raise ValueError(
                f'config {c.name!r} has {len(c.seeds)} seeds, n_repeats is {n_repeats}'
            )


def _open_state(
    path: Path,
    configs: Sequence[Config],
    head: dict[str, Any],
    targets: list[int],
    base_seed: int,
    resume: bool,
) -> dict[str, Any]:
    """Start a new state, or load the old one and check the caller's sweep still matches it."""
    if resume and not path.exists():
        raise ValueError(f'resume=True but there is no state file at {path}')
    if not resume and path.exists():
        raise ValueError(f'{path} already exists, pass resume=True to continue that sweep')

    if resume:
        state = load_state(path)
        for key, want in head.items():
            if state[key] != want:
                raise ValueError(f'{key} is {want!r} but the state at {path} has {state[key]!r}')
        missing = sorted(state['configs'].keys() - {c.name for c in configs})
        if missing:
            raise ValueError(f'configs {missing} are in the state but missing from configs')
    else:
        # a fresh launch would restore these into its new trials as if they were its own
        for c in configs:
            for r in range(head['n_repeats']):
                run_dir = path.parent / c.name / f'repeat{r}'
                stale = [*run_dir.glob(LATEST_CKPT), *sorted(run_dir.glob('rung*.pth'))]
                if stale:
                    raise ValueError(
                        f'{stale[0]} is left from an earlier sweep, pass resume=True to continue '
                        'it or move the old files away'
                    )
        state = {'version': STATE_VERSION, **head, 'targets': targets, 'rung': 0, 'configs': {}}

    for c in configs:
        # the round trip makes tuples equal lists, as they will be once stored
        params = json.loads(json.dumps(c.params))
        seeds = [
            c.seeds[r] if c.seeds is not None else config_seed(base_seed, head['sweep'], c.name, r)
            for r in range(head['n_repeats'])
        ]
        entry = state['configs'].get(c.name)
        if entry is None:
            if resume and head['halving'] is not None:
                raise ValueError(
                    f'config {c.name!r} is new, configs can only be added to a flat run'
                )
            repeats = [{'seed': s, 'run_id': None, 'scores': []} for s in seeds]
            state['configs'][c.name] = {'params': params, 'pruned_at': None, 'repeats': repeats}
            continue

        old = entry['params']
        diff = sorted(
            k for k in params.keys() | old.keys() if params.get(k, _MISSING) != old.get(k, _MISSING)
        )
        if diff:
            raise ValueError(f'config {c.name!r}: params differ from the state on keys {diff}')
        for r, s in enumerate(seeds):
            stored = entry['repeats'][r]['seed']
            if s != stored:
                raise ValueError(
                    f'config {c.name!r} repeat {r}: seed is {s} but the state has {stored}, '
                    'base_seed or seeds changed since launch'
                )

    write_json_atomic(path, state)
    return state


def _run_segment(
    state: dict[str, Any],
    state_path: Path,
    cfg: Config,
    r: int,
    i: int,
    run_dir: Path,
    trial_factory: Callable[[], Trial],
    logger: RunLogger | None,
    score: Score | None,
) -> None:
    """Train one (config, repeat) to rung i's target and commit it, or skip it when done."""
    entry = state['configs'][cfg.name]['repeats'][r]
    t = state['targets'][i]
    p = state['targets'][i - 1] if i > 0 else 0
    latest = run_dir / LATEST_CKPT
    rung_ckpt = run_dir / f'rung{i}.pth'
    where = f'config {cfg.name!r} repeat {r}'

    # done means score and rung file both landed, a latest left next to them is stale
    if len(entry['scores']) > i and rung_ckpt.exists():
        latest.unlink(missing_ok=True)
        return

    seed_everything(entry['seed'])
    run_dir.mkdir(parents=True, exist_ok=True)
    trial = trial_factory()
    failed = True
    try:
        trial.setup(cfg.params, entry['seed'], run_dir)

        from_latest = latest.exists()
        if from_latest:
            trial.restore(latest)
            _check_epoch(trial.epoch, p, t, f'{where}: epoch after restoring {latest.name}')

        # latest at the target with no score means the kill hit before the score write
        # that history is gone and a zero epoch train_until can't score, so redo the segment
        lost = score is not None and len(entry['scores']) <= i and trial.epoch == t
        if from_latest and lost:
            msg = f'{where}: {latest.name} at epoch {t} has no stored score, retraining from {p}'
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            trial.teardown()
            latest.unlink()
            seed_everything(entry['seed'])
            trial = trial_factory()
            trial.setup(cfg.params, entry['seed'], run_dir)
            from_latest = False

        if not from_latest and i > 0:
            prev = run_dir / f'rung{i - 1}.pth'
            if not prev.exists():
                raise FileNotFoundError(f'{where}: no {latest.name} and no {prev} to resume from')
            trial.restore(prev)
            _check_epoch(trial.epoch, p, p, f'{where}: epoch after restoring {prev.name}')
        elif not from_latest:
            _check_epoch(trial.epoch, 0, 0, f'{where}: epoch after setup')

        if logger is not None:
            name = f'{cfg.name}-repeat{r}'
            run_id = logger.init(name, tags=cfg.tags, config=cfg.params, run_id=entry['run_id'])
            # stored at once so a crash later in the segment still resumes the same run
            if run_id != entry['run_id']:
                entry['run_id'] = run_id
                write_json_atomic(state_path, state)

        # a stored score means the last run died after committing it but before the rung file
        if len(entry['scores']) > i:
            _check_epoch(trial.epoch, t, t, f'{where}: epoch with the score already stored')
        else:
            history = trial.train_until(t, logger, latest)
            _check_epoch(trial.epoch, t, t, f'{where}: epoch after train_until({t})')
            entry['scores'].append(None if score is None else score_history(history, score))
            write_json_atomic(state_path, state)

        # torch.save is not atomic, so the rung file only appears once whole
        tmp = rung_ckpt.with_name(rung_ckpt.name + '.tmp')
        trial.save(tmp)
        os.replace(tmp, rung_ckpt)
        latest.unlink(missing_ok=True)
        _log.info(f'committed {where} rung {i} at epoch {t}, score {entry["scores"][i]}')
        failed = False
    finally:
        # on the error path a cleanup failure only warns, so the first error is what raises
        cleanup = [trial.teardown] if logger is None else [logger.finish, trial.teardown]
        first_err = None
        for step in cleanup:
            try:
                step()
            except Exception as e:
                if not failed:
                    first_err = first_err or e
                else:
                    msg = f'{where}: {step.__name__} raised {e!r} while handling an earlier error'
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if first_err is not None:
            raise first_err


def _prune(state: dict[str, Any], state_path: Path, i: int, halving: Halving, sense: str) -> None:
    """Cut configs after rung i, IQR fence first and then the survival cut, and advance rung."""
    rung_scores = {}
    for name, entry in state['configs'].items():
        if entry['pruned_at'] is None:
            per_repeat = [rep['scores'][i] for rep in entry['repeats']]
            rung_scores[name] = sum(per_repeat) / len(per_repeat)
            if math.isnan(rung_scores[name]):
                msg = (
                    f'config {name!r} has a NaN score at rung {i}, it ranks after every finite one'
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

    fenced = []
    if halving.outlier_fence is not None:
        fenced = outliers(rung_scores, halving.outlier_fence, sense)
    rest = {name: s for name, s in rung_scores.items() if name not in fenced}
    kept = survivors(rest, halving.survival_fraction, sense)
    cut = sorted(rung_scores.keys() - set(kept))
    for name in cut:
        state['configs'][name]['pruned_at'] = i

    # cuts and rung land in one write, so a resume between rungs never prunes twice or skips it
    state['rung'] = i + 1 if len(kept) > 1 else len(state['targets'])
    write_json_atomic(state_path, state)
    stop = ', one left so the sweep stops' if len(kept) == 1 else ''
    _log.info(f'rung {i} prune kept {kept}, cut {cut}, of those by the IQR fence {fenced}{stop}')


def _check_epoch(epoch: int, lo: int, hi: int, what: str) -> None:
    """Raise RuntimeError unless lo <= epoch <= hi, what names the config, repeat, and step."""
    if lo <= epoch <= hi:
        return
    want = f'{lo}' if lo == hi else f'{lo} <= epoch <= {hi}'
    raise RuntimeError(f'{what} is {epoch}, expected {want}')
