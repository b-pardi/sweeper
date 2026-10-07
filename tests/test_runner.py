import fcntl
import json
import logging
import math
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import pytest

from sweeper import halving
from sweeper.config import Config
from sweeper.halving import Halving, Score
from sweeper.runlog import RunLogger
from sweeper.runner import run_sweep
from sweeper.seeds import config_seed

A = Config('a', {'lr': 0.1})
B = Config('b', {'lr': 0.2, 'depth': (1, 2)})  # a tuple comes back as a list from the state
RUN = {'sweep': 's', 'base_seed': 7}
SEED_A = config_seed(7, 's', 'a', 0)
SEED_B = config_seed(7, 's', 'b', 0)


def _bump_seed(state: dict) -> None:
    state['configs']['a']['repeats'][0]['seed'] += 1


def _boom(*args) -> None:
    raise OSError('injected replace failure')


@pytest.mark.parametrize(
    'keys, direction, err, match, kept, scores',
    [
        (('loss',), lambda k: 'min', None, None, 'a', {'a': [1.5, 1.5], 'b': [4.5]}),
        (('loss',), lambda k: 'max', None, None, 'b', {'a': [2.5], 'b': [5.5, 5.5]}),
        (('nope',), {'loss': 'min'}.__getitem__, KeyError, 'nope', None, None),
        (('loss',), lambda k: True, ValueError, "'max', got True", None, None),
        (('loss', 'acc'), dict(loss='min', acc='max').get, ValueError, 'one dir', None, None),
    ],
)
def test_score_direction_comes_from_callable(
    tmp_path: Path, fake_trial, keys, direction, err, match, kept, scores
) -> None:
    """Both the top-k pick and the survival ranking follow score.direction."""
    # rung 0 sees 3 epochs, top_fraction 0.5 keeps the best 2
    fake_trial.histories = {'a': {'loss': [3.0, 1.0, 2.0] * 2}, 'b': {'loss': [5.0, 4.0, 6.0] * 2}}
    args = {'epochs': 6, 'halving': Halving(rung_epochs=3, survival_fraction=0.5), **RUN}
    if err is not None:
        with pytest.raises(err, match=match):
            run_sweep([A, B], fake_trial, tmp_path, score=Score(keys, direction, 0.5), **args)
        assert fake_trial.calls == []
        return

    state = run_sweep([A, B], fake_trial, tmp_path, score=Score(keys, direction, 0.5), **args)
    cut = 'b' if kept == 'a' else 'a'
    assert state['configs'][kept]['pruned_at'] is None
    assert state['configs'][cut]['pruned_at'] == 0
    # the survivor trains rung 1 on its own, the cut config keeps only its rung 0 score
    assert {n: c['repeats'][0]['scores'] for n, c in state['configs'].items()} == scores


# what a halving row expects unless it says otherwise, epochs 4 in 2 rungs, nothing cut
PLAIN = {
    'halving': Halving(rung_epochs=2, survival_fraction=1.0),
    'score': True,
    'sense': 'min',
    'epochs': 4,
    'n_repeats': 1,
    'min_prune': None,  # patched over the IQR prune's minimum config count
    'err': None,
    'targets': [2, 4],
    'pruned': {},  # config -> pruned_at, the rest are never cut
    'fenced': [],
}
ABCDE = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 100}
NAN = float('nan')
NAN_MID = {'a': 1, 'b': 2, 'c': NAN, 'd': 3, 'e': 4}  # 4 finite scores, so the fence runs


@pytest.mark.parametrize(
    'scores, case',
    [
        pytest.param(
            {'a': 1, 'b': 2, 'c': 3},
            {'halving': None, 'score': False, 'n_repeats': 2, 'targets': [4]},
            id='flat',
        ),
        pytest.param(
            {'a': 1, 'b': 2},
            {'epochs': 5, 'targets': [2, 4, 5]},
            id='targets-tail',
        ),
        pytest.param(
            {'a': 1, 'b': 2},
            {'halving': Halving(rung_epochs=7, survival_fraction=0.5), 'targets': [4]},
            id='targets-one-rung',
        ),
        pytest.param(
            {'a': 1, 'b': 2, 'c': 3},
            {'halving': Halving(2, 0.5), 'n_repeats': 2, 'pruned': {'c': 0}},
            id='n3-f0.5',
        ),
        pytest.param({'a': 1}, {'halving': Halving(2, 0.5)}, id='n1-f0.5'),
        pytest.param(
            {'a': 1},
            {'halving': Halving(2, 1.0), 'epochs': 5, 'targets': [2, 4, 5]},
            id='n1-f1.0',
        ),
        pytest.param(
            {'a': 1, 'b': 2, 'c': 3, 'd': 4},
            {'halving': Halving(2, 0.1), 'pruned': {'b': 0, 'c': 0, 'd': 0}},
            id='n4-f0.1',
        ),
        pytest.param(
            {'b': 1, 'a': 1},
            {'halving': Halving(2, 0.5), 'pruned': {'b': 0}},
            id='tie-min',
        ),
        pytest.param(
            {'b': 1, 'a': 1},
            {'halving': Halving(2, 0.5), 'sense': 'max', 'pruned': {'b': 0}},
            id='tie-max',
        ),
        pytest.param({'a': 1}, {'score': False, 'err': 'needs a Score'}, id='no-score'),
        pytest.param(
            ABCDE,
            {
                'halving': Halving(2, 0.5, outlier_fence=1.5),
                'pruned': {'c': 0, 'd': 0, 'e': 0},
                'fenced': ['e'],
            },
            id='fence-min',
        ),
        pytest.param(
            {'a': 100, 'b': 99, 'c': 98, 'd': 97, 'e': 1},
            {
                'halving': Halving(2, 1.0, outlier_fence=1.5),
                'sense': 'max',
                'pruned': {'e': 0},
                'fenced': ['e'],
            },
            id='fence-max',
        ),
        pytest.param(
            ABCDE,
            {'halving': Halving(2, 1.0, outlier_fence=1.5), 'min_prune': 6},
            id='fence-below-min',
        ),
        pytest.param(ABCDE, {}, id='no-fence'),
        pytest.param(NAN_MID, {'halving': Halving(2, 0.8), 'pruned': {'c': 0}}, id='nan-min'),
        pytest.param(
            NAN_MID,
            {'halving': Halving(2, 0.8), 'sense': 'max', 'pruned': {'c': 0}},
            id='nan-max',
        ),
        pytest.param(
            NAN_MID,
            {'halving': Halving(2, 1.0, outlier_fence=1.5), 'pruned': {'c': 0}, 'fenced': ['c']},
            id='nan-fence-min',
        ),
        pytest.param(
            NAN_MID,
            {
                'halving': Halving(2, 1.0, outlier_fence=1.5),
                'sense': 'max',
                'pruned': {'c': 0},
                'fenced': ['c'],
            },
            id='nan-fence-max',
        ),
    ],
)
def test_halving_is_opt_in_only(
    tmp_path: Path, fake_trial, monkeypatch, caplog, scores: dict, case: dict
) -> None:
    """Halving is opt-in, cuts by rank after the IQR fence, and survivors train to epochs."""
    c = {**PLAIN, **case}
    if c['min_prune'] is not None:
        monkeypatch.setattr(halving, 'MIN_PRUNE_CONFIGS', c['min_prune'])
    configs = [Config(n, {}) for n in scores]
    fake_trial.histories = {n: {'loss': [float(v)] * c['epochs']} for n, v in scores.items()}
    score = Score(('loss',), lambda k: c['sense'], 1.0) if c['score'] else None
    args = {'epochs': c['epochs'], 'n_repeats': c['n_repeats'], 'halving': c['halving'], **RUN}
    out = tmp_path / 'out'
    if c['err'] is not None:
        with pytest.raises(ValueError, match=c['err']):
            run_sweep(configs, fake_trial, out, score=score, **args)
        assert fake_trial.calls == []
        return

    with caplog.at_level(logging.INFO, logger='sweeper'), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        state = run_sweep(configs, fake_trial, out, score=score, **args)
    nan = [n for n, v in scores.items() if math.isnan(v)]
    assert [str(x.message) for x in w] == [
        f'config {n!r} has a NaN score at rung 0, it ranks after every finite one' for n in nan
    ]
    targets = c['targets']
    assert state['targets'] == targets
    assert state['rung'] == len(targets) - 1
    for name in scores:
        pruned_at = c['pruned'].get(name)
        assert state['configs'][name]['pruned_at'] == pruned_at
        last = len(targets) - 1 if pruned_at is None else pruned_at
        for r in range(c['n_repeats']):
            run_dir = out / name / f'repeat{r}'
            assert sorted(p.name for p in run_dir.glob('rung*.pth')) == [
                f'rung{j}.pth' for j in range(last + 1)
            ]
            assert json.loads((run_dir / f'rung{last}.pth').read_text())['epoch'] == targets[last]
            assert len(state['configs'][name]['repeats'][r]['scores']) == last + 1

    prunes = [rec.getMessage() for rec in caplog.records if 'prune' in rec.getMessage()]
    # a lone config has nothing to prune, so it logs no prune line
    assert len(prunes) == (len(targets) - 1 if len(scores) > 1 else 0)
    if prunes:
        assert f'IQR fence {c["fenced"]}' in prunes[0]

    # a finished sweep does nothing on resume
    fake_trial.calls.clear()
    again = run_sweep(configs, fake_trial, out, score=score, resume=True, **args)
    # through JSON, since NaN != NaN would fail a plain dict compare
    assert json.dumps(again) == json.dumps(state)
    assert fake_trial.calls == []


@pytest.mark.parametrize(
    'first, second, edit, broken_replace, err, match',
    [
        ([A, B], [B, A], None, False, None, None),  # reorder reruns nothing
        ([A], [A, B], None, False, None, None),  # appended config trains
        ([A], [Config('a', {'lr': 0.3})], None, False, ValueError, r"keys \['lr'\]"),
        ([A, B], [A], None, False, ValueError, "configs \\['b'\\] are in the state"),
        ([A], [A], lambda s: s.update(version=2), False, ValueError, 'version must be 1, got 2'),
        ([A], [A], lambda s: s.pop('version'), False, ValueError, 'got None'),
        ([A], [A], _bump_seed, False, ValueError, 'seed is'),
        (None, [A, A], None, False, ValueError, 'duplicate'),
        (None, [Config('../x', {})], None, False, ValueError, 'filesystem safe'),
        (None, [Config('..', {})], None, False, ValueError, 'filesystem safe'),
        (None, [Config('.hidden', {})], None, False, ValueError, 'filesystem safe'),
        (None, [Config('a/b', {})], None, False, ValueError, 'filesystem safe'),
        ([A], [A, B], None, True, OSError, 'injected replace'),
        ([A], [A], None, False, RuntimeError, r'sweep\.lock is held'),
    ],
)
def test_state_file_keyed_by_name_with_version(
    tmp_path: Path, fake_trial, monkeypatch, first, second, edit, broken_replace, err, match
) -> None:
    """State is keyed by config name, and a failed resume leaves the state file untouched."""
    out = tmp_path / 'out'
    state_path = out / 'sweep_state.json'
    if first is not None:
        run_sweep(first, fake_trial, out, epochs=2, **RUN)
        assert set(json.loads(state_path.read_text())['configs']) == {c.name for c in first}
    if edit is not None:
        state = json.loads(state_path.read_text())
        edit(state)
        state_path.write_text(json.dumps(state))
    before = state_path.read_text() if first is not None else None
    if broken_replace:
        monkeypatch.setattr(os, 'replace', _boom)
    fake_trial.calls.clear()

    if err is not None:
        # the one RuntimeError row runs while the test holds the out_dir lock
        lock = open(out / 'sweep.lock', 'a') if err is RuntimeError else nullcontext()
        with lock as held, pytest.raises(err, match=match):
            if held is not None:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_sweep(second, fake_trial, out, epochs=2, resume=first is not None, **RUN)
        assert fake_trial.calls == []
        if first is not None:
            assert state_path.read_text() == before
            assert not list(out.glob('*.tmp'))
        return

    state = run_sweep(second, fake_trial, out, epochs=2, resume=True, **RUN)
    old = json.loads(before)['configs']
    new_names = [c.name for c in second if c.name not in old]
    assert [c[1] for c in fake_trial.calls if c[0] == 'setup'] == new_names
    assert set(state['configs']) == {c.name for c in second}
    for name in old:
        assert state['configs'][name] == old[name]  # seeds, scores, and run ids kept
    assert json.loads(state_path.read_text()) == state


@pytest.mark.parametrize(
    'knobs, overrides, err, match',
    [
        ({}, {}, None, None),
        ({'restore_shift': 3}, {}, RuntimeError, r"'b' repeat 0: epoch after restoring .* is 5"),
        ({'restore_shift': -3}, {}, RuntimeError, r'is -1, expected 0 <= epoch <= 4'),
        ({'short_by': 1}, {}, RuntimeError, r"'b' repeat 0: epoch after train_until\(4\) is 3"),
        ({}, {'resume': False}, ValueError, 'already exists'),
        ({}, {'out_dir': 'empty'}, ValueError, 'no state file'),
        ({}, {'epochs': 5}, ValueError, 'epochs is 5 but the state .* has 4'),
        (
            {},
            {'score': Score(('loss',), lambda k: 'min', 0.25)},
            ValueError,
            r"'top_fraction': 0\.25.* but the state .* has .*'top_fraction': 0\.5",
        ),
        (
            {},
            {'resume': False, 'drop_state': True},
            ValueError,
            r'a/repeat0/rung0\.pth is left from an earlier sweep, pass resume=True',
        ),
    ],
)
def test_resume_skips_done_and_restores_epoch(
    tmp_path: Path, fake_trial, knobs: dict, overrides: dict, err, match
) -> None:
    """After b crashes at epoch 2, a resume leaves a alone and trains b from its latest."""
    fake_trial.crash = {'b': 2}
    fake_trial.histories = {n: {'loss': [4.0, 3.0, 2.0, 1.0]} for n in 'ab'}
    score = Score(('loss',), lambda k: 'min', 0.5)
    with pytest.raises(RuntimeError, match='injected train_until'):
        run_sweep([A, B], fake_trial, tmp_path / 'out', epochs=4, score=score, **RUN)
    fake_trial.crash = {}
    fake_trial.calls.clear()
    for k, v in knobs.items():
        setattr(fake_trial, k, v)

    args = {'out_dir': 'out', 'epochs': 4, 'resume': True, 'score': score, **RUN, **overrides}
    args['out_dir'] = tmp_path / args['out_dir']
    # drop_state fakes a fresh launch into a dir whose checkpoints outlived their state file
    if args.pop('drop_state', False):
        (args['out_dir'] / 'sweep_state.json').unlink()
    if err is not None:
        with pytest.raises(err, match=match):
            run_sweep([A, B], fake_trial, **args)
        if err is ValueError:
            assert fake_trial.calls == []
        return

    state = run_sweep([A, B], fake_trial, **args)
    assert fake_trial.calls == [
        ('setup', 'b', SEED_B),
        ('restore', 'latest.pth', 2),
        ('train_until', 2, 4),
        ('save', 'rung0.pth.tmp'),
        ('teardown',),
    ]
    assert [len(state['configs'][n]['repeats'][0]['scores']) for n in 'ab'] == [1, 1]
    assert not (tmp_path / 'out' / 'b' / 'repeat0' / 'latest.pth').exists()


SETUP = ('setup', 'a', SEED_A)
SAVE = ('save', 'rung0.pth.tmp')
DOWN = ('teardown',)
FULL = [SETUP, ('train_until', 0, 3), SAVE, DOWN]
FROM_1 = [SETUP, ('restore', 'latest.pth', 1), ('train_until', 1, 3), SAVE, DOWN]
FROM_3 = [SETUP, ('restore', 'latest.pth', 3), SAVE, DOWN]
CRASHED = [SETUP, ('train_until', 0, 3), DOWN]
# train_until at the target trains nothing and hands back the history latest kept
AT_3 = [SETUP, ('restore', 'latest.pth', 3), ('train_until', 3, 3)]
# halving run with targets [2, 3], a and b both survive rung 0
SAVE_1 = ('save', 'rung1.pth.tmp')
FROM_RUNG0 = [('restore', 'rung0.pth', 2), ('train_until', 2, 3)]
RUNG0_AB = [SETUP, ('train_until', 0, 2), SAVE, DOWN, ('setup', 'b', SEED_B)]
RUNG0_AB += [('train_until', 0, 2), SAVE, DOWN]
RUNG1_B = [('setup', 'b', SEED_B), *FROM_RUNG0, SAVE_1, DOWN]
# what a row expects unless it says otherwise, the plain clean run
CLEAN = {
    'configs': [A],
    'halving': None,
    'epochs': 3,
    'kill_at': None,
    'rung': 0,  # the rung whose commit the row checks
    'fail': set(),
    'crash': None,
    'plant_latest': False,
    'first_err': None,
    'first_warn': None,
    'first_calls': FULL,
    'first_scores': [1.5],
    'committed': True,
    'rerun_calls': [],
    'rerun_scores': [1.5],
}
# every rerun below scores what the unbroken run does, since latest keeps the partial history
TRAIN_CRASH = {
    'crash': 1,
    'first_err': 'injected train_until',
    'first_calls': CRASHED,
    'first_scores': [],
    'committed': False,
    'rerun_calls': FROM_1,
}


def _run_checked(err: str | None, warn: str | None, *args, **kwargs) -> None:
    """run_sweep expecting exactly this error and warning, any other warning fails."""
    raises = nullcontext() if err is None else pytest.raises(RuntimeError, match=err)
    warns = nullcontext() if warn is None else pytest.warns(RuntimeWarning, match=warn)
    with raises, warns, warnings.catch_warnings():
        warnings.simplefilter('error' if warn is None else 'always')
        run_sweep(*args, **kwargs)


@pytest.mark.parametrize(
    'case',
    [
        pytest.param({}, id='normal-order'),
        pytest.param({'plant_latest': True}, id='kill-after-rename'),
        pytest.param(
            {
                'fail': {'save'},
                'first_err': 'injected save',
                'committed': False,
                'rerun_calls': FROM_3,
            },
            id='save-crash',
        ),
        pytest.param(TRAIN_CRASH, id='train-crash'),
        pytest.param(
            {**TRAIN_CRASH, 'fail': {'teardown'}, 'first_warn': 'teardown raised'},
            id='teardown-error-path',
        ),
        pytest.param(
            {'fail': {'teardown'}, 'first_err': 'injected teardown'}, id='teardown-normal-path'
        ),
        pytest.param(
            {
                'fail': {'train_until'},
                'first_err': 'injected kill after train_until',
                'first_calls': CRASHED,
                'first_scores': [],
                'committed': False,
                'rerun_calls': [*AT_3, SAVE, DOWN],
            },
            id='kill-after-train-until',
        ),
        pytest.param(
            {
                'configs': [A, B],
                'halving': Halving(rung_epochs=2, survival_fraction=1.0),
                'kill_at': 3,
                'rung': 1,
                'fail': {'train_until'},
                'first_err': 'injected kill after train_until',
                'first_calls': [*RUNG0_AB, SETUP, *FROM_RUNG0, DOWN],
                'first_scores': [1.0],
                'committed': False,
                'rerun_calls': [*AT_3, SAVE_1, DOWN, *RUNG1_B],
                'rerun_scores': [1.0, 2.0],  # rung 0 scores [3, 1] -> 1, rung 1 scores [2]
            },
            id='kill-after-train-until-rung1',
        ),
        pytest.param(
            {
                'halving': Halving(rung_epochs=2, survival_fraction=1.0),
                'epochs': 4,
                'rung': 1,
                'crash': 3,
                'first_err': 'injected train_until',
                'first_calls': [SETUP, ('train_until', 0, 2), SAVE, DOWN, SETUP]
                + [('restore', 'rung0.pth', 2), ('train_until', 2, 4), DOWN],
                'first_scores': [1.0],
                'committed': False,
                'rerun_calls': [SETUP, ('restore', 'latest.pth', 3), ('train_until', 3, 4)]
                + [SAVE_1, DOWN],
                # rung 1 scores [2, 4] -> 2, epoch 4 alone would give 4
                'rerun_scores': [1.0, 2.0],
            },
            id='train-crash-mid-rung1',
        ),
    ],
)
def test_trial_lifecycle_order_and_commit(tmp_path: Path, fake_trial, fake_wandb, case) -> None:
    """The score commits before the rung file, and every failure path resumes cleanly."""
    c = {**CLEAN, **case}
    out = tmp_path / 'out'
    run_dir = out / 'a' / 'repeat0'
    score = Score(('loss',), lambda k: 'min', 0.5)
    kwargs = {'epochs': c['epochs'], 'logger': RunLogger('proj'), 'score': score, **RUN}
    kwargs['halving'] = c['halving']
    rung_ckpt = run_dir / f'rung{c["rung"]}.pth'
    # best half of the first 3 is [1, 2], epoch 4 is only reached with epochs 4
    fake_trial.histories = {n: {'loss': [3.0, 1.0, 2.0, 4.0]} for n in 'ab'}
    fake_trial.fail = c['fail']
    fake_trial.kill_at = c['kill_at']
    if c['crash'] is not None:
        fake_trial.crash = {'a': c['crash']}

    _run_checked(c['first_err'], c['first_warn'], c['configs'], fake_trial, out, **kwargs)
    assert fake_trial.calls == c['first_calls']
    assert all(run.finished for run in fake_wandb.runs)

    state = json.loads((out / 'sweep_state.json').read_text())
    repeat = state['configs']['a']['repeats'][0]
    assert repeat['run_id'] == 'fake-run-id'
    assert repeat['scores'] == c['first_scores']
    assert rung_ckpt.exists() == c['committed']
    assert (run_dir / 'latest.pth').exists() != c['committed']

    if c['plant_latest']:
        (run_dir / 'latest.pth').write_text(json.dumps({'epoch': 3, 'payload': 'a'}))
    fake_trial.fail = set()
    fake_trial.crash = {}
    fake_trial.calls.clear()
    # any warning on the rerun fails the row
    _run_checked(None, None, c['configs'], fake_trial, out, resume=True, **kwargs)
    assert fake_trial.calls == c['rerun_calls']

    state = json.loads((out / 'sweep_state.json').read_text())
    assert state['configs']['a']['repeats'][0]['scores'] == c['rerun_scores']
    assert json.loads(rung_ckpt.read_text())['epoch'] == c['epochs']
    assert not (run_dir / 'latest.pth').exists()
    assert not list(run_dir.glob('*.tmp'))
    # only each config's first logger run is new, every later one reuses the stored id
    n_new = len(c['configs'])
    assert all('id' not in kw for kw in fake_wandb.init_kwargs[:n_new])
    assert all(kw['id'] == 'fake-run-id' for kw in fake_wandb.init_kwargs[n_new:])
