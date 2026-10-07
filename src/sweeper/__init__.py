"""Crash-safe runner for a list of configs times repeats.

Every finished segment is committed to disk, so a killed sweep resumes where it stopped.
Successive halving is opt-in.
"""

from sweeper.config import Config, explicit_flags, load_sweep, resolve_params
from sweeper.halving import Halving, Score
from sweeper.runlog import RunLogger
from sweeper.runner import run_sweep
from sweeper.seeds import (
    config_seed,
    derive_seed,
    get_rng_states,
    seed_everything,
    set_rng_states,
)
from sweeper.trial import Trial

__version__ = '0.1.0'

__all__ = [
    # running a sweep
    'run_sweep',
    'Trial',
    # what to run and how to rank it
    'Config',
    'Score',
    'Halving',
    # sweep files and params
    'load_sweep',
    'resolve_params',
    'explicit_flags',
    # logging
    'RunLogger',
    # seeds and rng state
    'derive_seed',
    'config_seed',
    'seed_everything',
    'get_rng_states',
    'set_rng_states',
]
