"""Seed derivation and global RNG helpers for the sweeper.

Seeds come from a hash of the base seed and string tokens, so the same (sweep, config, repeat)
always gets the same seed no matter the run order.
"""

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch

SEED_MOD = 2**31 - 1  # mersenne prime, not 2**31, changing it changes every derived seed


def derive_seed(base: int, *tokens: str) -> int:
    """Derive a seed from a base seed and string tokens.

    Args:
        base: Base seed.
        *tokens: Strings mixed into the hash in order, each goes through str().

    Returns:
        An int in [0, 2**31 - 2].
    """
    h = hashlib.sha256()
    h.update(str(base).encode())
    for t in tokens:
        h.update(('::' + str(t)).encode())
    return int(h.hexdigest()[:16], 16) % SEED_MOD


def config_seed(base_seed: int, sweep: str, config: str, repeat: int) -> int:
    """Seed for one repeat of one config in a sweep.

    Args:
        base_seed: Base seed of the whole run.
        sweep: Sweep name.
        config: Config name.
        repeat: Repeat index.

    Returns:
        The derived seed.
    """
    return derive_seed(base_seed, sweep, config, f'repeat{repeat}')


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, and torch (cpu and every cuda device).

    Also sets PYTHONHASHSEED and, with deterministic, CUBLAS_WORKSPACE_CONFIG in os.environ.

    Args:
        seed: Seed for all three libraries.
        deterministic: Force deterministic cudnn and torch algorithms, slower on gpu.
    """
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def get_rng_states() -> dict[str, Any]:
    """Snapshot the python, numpy, and torch RNG states for a checkpoint.

    Returns:
        Dict with keys py, np, torch_cpu, and torch_cuda (None without cuda).
    """
    return {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_states(states: dict[str, Any]) -> None:
    """Restore the RNG states saved by get_rng_states.

    A missing or None key is skipped, so a cpu-only checkpoint loads on a gpu box.

    Args:
        states: Dict from get_rng_states, possibly after a torch.save and torch.load.

    Raises:
        TypeError: If states is not a dict.
    """
    if not isinstance(states, dict):
        raise TypeError(f'states must be a dict from get_rng_states, got {type(states).__name__}')

    if states.get('py') is not None:
        random.setstate(states['py'])

    if states.get('np') is not None:
        np.random.set_state(states['np'])

    if states.get('torch_cpu') is not None:
        torch.set_rng_state(_to_byte_tensor(states['torch_cpu']))

    if states.get('torch_cuda') is not None and torch.cuda.is_available():
        cuda_states = states['torch_cuda']
        if torch.is_tensor(cuda_states):
            cuda_states = [cuda_states]
        torch.cuda.set_rng_state_all([_to_byte_tensor(s) for s in cuda_states])


def _to_byte_tensor(x: Any) -> torch.Tensor:
    """Flat cpu uint8 tensor from a state that may be a tensor, list, or 1-tuple of one."""
    if isinstance(x, (list, tuple)) and len(x) == 1 and torch.is_tensor(x[0]):
        x = x[0]

    if torch.is_tensor(x):
        return x.detach().to(device='cpu', dtype=torch.uint8).flatten()

    return torch.as_tensor(x, dtype=torch.uint8, device='cpu').flatten()
