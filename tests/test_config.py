import argparse
import json
from pathlib import Path

import pytest

from sweeper.config import explicit_flags, load_sweep, resolve_params

DEFAULTS = {'lr': 0.1, 'seed': 42}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('-s', '--seed', type=int, default=42)
    p.add_argument('--fast', action='store_true')
    p.add_argument('--sizes', nargs='+', type=int, default=[1])
    return p


# cli_keys None means use whatever explicit_flags finds, want is None when KeyError is expected
@pytest.mark.parametrize(
    'argv, base, overrides, cli_keys, want_flags, want',
    [
        ([], {}, {}, None, set(), {'lr': 0.1, 'seed': 42}),
        ([], {'lr': 0.5}, {}, None, set(), {'lr': 0.5, 'seed': 42}),
        (['--lr', '0.9'], {'lr': 0.5}, {}, None, {'lr'}, {'lr': 0.9, 'seed': 42}),
        (['--lr', '0.9'], {'lr': 0.5}, {}, set(), {'lr'}, {'lr': 0.5, 'seed': 42}),
        (['--lr', '0.9'], {}, {'lr': 0.3}, None, {'lr'}, {'lr': 0.3, 'seed': 42}),
        (['--fast'], {}, {}, None, {'fast'}, {'lr': 0.1, 'seed': 42, 'fast': True}),
        (['--seed', '42'], {'seed': 1}, {}, None, {'seed'}, {'lr': 0.1, 'seed': 42}),
        (['-s', '7'], {}, {}, None, {'seed'}, {'lr': 0.1, 'seed': 7}),
        (['--sizes', '2', '3'], {}, {}, None, {'sizes'}, {'lr': 0.1, 'seed': 42, 'sizes': [2, 3]}),
        (['--bogus', '1'], {}, {}, None, set(), {'lr': 0.1, 'seed': 42}),
        ([], {}, {}, {'nope'}, set(), None),
        (['sub'], {}, {}, None, TypeError, None),
    ],
)
def test_params_precedence_and_cli_keys(argv, base, overrides, cli_keys, want_flags, want):
    """Layers go defaults, base_params, explicit CLI, overrides. Flags come from argv alone."""
    parser = _parser()
    # a subcommand's defaults would count as explicit, so subparsers are refused
    if want_flags is TypeError:
        parser.add_subparsers(dest='cmd').add_parser('sub').add_argument('--depth', default=2)
        with pytest.raises(TypeError, match='subparsers'):
            explicit_flags(parser, argv)
        return
    assert explicit_flags(parser, argv) == want_flags

    cli = vars(parser.parse_known_args(argv)[0])
    keys = want_flags if cli_keys is None else cli_keys
    if want is None:
        with pytest.raises(KeyError, match='nope'):
            resolve_params(DEFAULTS, base, cli, keys, overrides)
        return
    assert resolve_params(DEFAULTS, base, cli, keys, overrides) == want


def _sweep(**over) -> dict:
    d = {'version': 1, 'epochs': 10, 'configs': [{'name': 'a', 'params': {}}]}
    d.update(over)
    return {k: v for k, v in d.items() if v is not None}


@pytest.mark.parametrize(
    'sweep, key',
    [
        (_sweep(extra=1), 'extra'),
        (_sweep(configs=[{'name': 'a', 'params': {}, 'bogus': 1}]), 'bogus'),
        (_sweep(_comment='hi'), '_comment'),
        (_sweep(epochs=None), 'epochs'),
        (_sweep(version=2), 'version'),
    ],
)
def test_sweep_file_unknown_keys_raise(tmp_path: Path, sweep: dict, key: str) -> None:
    """Bad key sets and a wrong version raise ValueError that names the offender."""
    path = tmp_path / 'sweep.json'
    path.write_text(json.dumps(sweep), encoding='utf-8')
    with pytest.raises(ValueError, match=key):
        load_sweep(path)
