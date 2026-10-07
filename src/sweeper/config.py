"""Sweep file loading and param layering for the sweeper.

The sweep file is plain JSON. params and base_params are opaque here, the caller validates them.
"""

import argparse
import copy
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SWEEP_VERSION = 1

_TOP_REQUIRED = {'version', 'epochs', 'configs'}
_TOP_ALLOWED = _TOP_REQUIRED | {'n_repeats', 'base_params', 'description'}
_CONFIG_REQUIRED = {'name', 'params'}
_CONFIG_ALLOWED = _CONFIG_REQUIRED | {'seeds', 'tags', 'description'}


@dataclass(frozen=True)
class Config:
    """One config of a sweep, what run_sweep takes."""

    name: str  # unique in the sweep, also the dir name
    params: dict[str, Any]  # per-config overrides, resolved by the caller
    seeds: tuple[int, ...] | None = None  # one per repeat, None derives them from the base seed
    tags: tuple[str, ...] = ()  # passed to the logger


def load_sweep(path: Path) -> dict[str, Any]:
    """Read a sweep file and check its version and key sets.

    Args:
        path: JSON sweep file.

    Returns:
        The file's dict, with n_repeats (1) and base_params ({}) filled in when absent.

    Raises:
        ValueError: on a wrong version, a missing required key, or any key outside the allowed
            set, at the top level or in a config.
    """
    with open(path, encoding='utf-8') as f:
        sweep = json.load(f)
    if not isinstance(sweep, dict):
        raise ValueError(f'{path}: sweep file must hold a JSON object, got {type(sweep).__name__}')

    _check_keys(sweep, _TOP_REQUIRED, _TOP_ALLOWED, f'{path} top level')
    if sweep['version'] != SWEEP_VERSION:
        raise ValueError(f'{path}: version must be {SWEEP_VERSION}, got {sweep["version"]!r}')
    if not isinstance(sweep['configs'], list):
        raise ValueError(f'{path}: configs must be a list, got {type(sweep["configs"]).__name__}')

    for i, cfg in enumerate(sweep['configs']):
        if not isinstance(cfg, dict):
            raise ValueError(f'{path}: configs[{i}] must be an object, got {type(cfg).__name__}')
        _check_keys(cfg, _CONFIG_REQUIRED, _CONFIG_ALLOWED, f'{path} configs[{i}]')

    sweep.setdefault('n_repeats', 1)
    sweep.setdefault('base_params', {})
    return sweep


def _check_keys(d: dict, required: set[str], allowed: set[str], where: str) -> None:
    """Raise if d is missing a required key or holds a key outside the allowed set."""
    missing = sorted(required - d.keys())
    if missing:
        raise ValueError(
            f'{where}: missing required keys {missing}, required are {sorted(required)}'
        )
    extra = sorted(d.keys() - allowed)
    if extra:
        raise ValueError(f'{where}: unknown keys {extra}, allowed are {sorted(allowed)}')


def resolve_params(
    defaults: Mapping[str, Any],
    base_params: Mapping[str, Any],
    cli: Mapping[str, Any],
    cli_keys: Collection[str],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Layer defaults, base_params, explicit CLI values, and per-config overrides, low to high.

    Args:
        defaults: the domain's default params.
        base_params: sweep-wide params from the sweep file.
        cli: all parsed CLI values.
        cli_keys: keys of cli that were explicitly given, only these are layered in.
        overrides: one config's params, these beat even an explicit CLI value.

    Returns:
        A new dict, the inputs are not touched.

    Raises:
        KeyError: if a key of cli_keys is missing from cli.
    """
    out = dict(defaults)
    out.update(base_params)
    out.update({k: cli[k] for k in cli_keys})
    out.update(overrides)
    return out


def explicit_flags(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> set[str]:
    """Find which flags are actually in argv, whatever their defaults are.

    Args:
        parser: the caller's parser, left untouched.
        argv: arguments to parse, None means sys.argv[1:].

    Returns:
        The dests of the flags that appeared. Unknown flags are ignored here, the real parser
        rejects them.

    Raises:
        TypeError: if parser has subparsers, their defaults can't be suppressed from here.
    """
    if any(isinstance(a, argparse._SubParsersAction) for a in parser._actions):
        raise TypeError(f'explicit_flags does not support subparsers, got parser {parser.prog!r}')

    shadow = copy.deepcopy(parser)
    # a dest only lands in the namespace when its flag appears if every default is SUPPRESS
    # argparse has no public way to drop defaults or required, hence the private attrs
    for action in shadow._actions:
        action.default = argparse.SUPPRESS
        action.required = False
    shadow._defaults = {}

    ns, _ = shadow.parse_known_args(argv)
    return set(vars(ns))
