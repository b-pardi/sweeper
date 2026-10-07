import random

import numpy as np
import pytest
import torch

from sweeper.config import Config
from sweeper.runner import run_sweep
from sweeper.seeds import config_seed, derive_seed, get_rng_states, seed_everything, set_rng_states


def test_config_seed_and_explicit_override(tmp_path, fake_trial) -> None:
    """Golden ints pin the hash, config_seed moves with each input, and explicit seeds win."""
    assert derive_seed(0, 'sweep', 'config', 'repeat0', 'extra') == 786154327
    assert derive_seed(12345, 'alpha', 'beta', 'repeat2') == 1842381920
    assert derive_seed(7, 'sweep', 'config', 'repeat3') == 1158324556
    assert config_seed(7, 'sweep', 'config', 3) == 1158324556

    base = config_seed(7, 'sweep', 'config', 3)
    assert config_seed(7, 'sweep', 'config', 3) == base
    assert config_seed(7, 'other', 'config', 3) != base
    assert config_seed(7, 'sweep', 'other', 3) != base
    assert config_seed(7, 'sweep', 'config', 4) != base

    # repeat 0 of every config runs before any repeat 1
    run = {'sweep': 'sweep', 'epochs': 1, 'base_seed': 7, 'n_repeats': 2}
    configs = [Config('a', {}, seeds=(11, 22)), Config('b', {})]
    run_sweep(configs, fake_trial, tmp_path / 'ok', **run)
    assert [c for c in fake_trial.calls if c[0] == 'setup'] == [
        ('setup', 'a', 11),
        ('setup', 'b', config_seed(7, 'sweep', 'b', 0)),
        ('setup', 'a', 22),
        ('setup', 'b', config_seed(7, 'sweep', 'b', 1)),
    ]
    with pytest.raises(ValueError, match="'a' has 1 seeds, n_repeats is 2"):
        run_sweep([Config('a', {}, seeds=(11,))], fake_trial, tmp_path / 'bad', **run)


def _draws() -> tuple:
    """One draw from each of the three global streams."""
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_rng_states_round_trip(tmp_path) -> None:
    """Seeding and restoring replays all three streams, and a generator state survives disk."""
    seed_everything(5)
    first = _draws()
    seed_everything(5)
    assert _draws() == first

    seed_everything(9)
    states = get_rng_states()
    expected = _draws()
    _draws()
    set_rng_states(states)
    assert _draws() == expected

    gen = torch.Generator().manual_seed(3)
    path = tmp_path / 'gen.pt'
    torch.save(gen.get_state(), path)
    expected_perm = torch.randperm(10, generator=gen)
    gen.manual_seed(99)
    gen.set_state(torch.load(path))
    assert torch.equal(torch.randperm(10, generator=gen), expected_perm)

    with pytest.raises(TypeError):
        set_rng_states([1, 2, 3])
