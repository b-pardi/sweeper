"""The sweep state file and the atomic JSON writer behind every write to it."""

import json
import os
from pathlib import Path
from typing import Any

STATE_VERSION = 1
STATE_FILE = 'sweep_state.json'


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write obj as JSON so path holds either the old file or the whole new one, never half.

    Args:
        path: target file, its dir must exist.
        obj: anything json.dump takes.

    Raises:
        TypeError: if obj holds something JSON can't store. path is left as it was.
    """
    # tmp in the same dir so os.replace never crosses filesystems
    tmp = path.with_name(path.name + '.tmp')
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2)
            f.flush()
            # without fsync a power cut can leave an empty file after the rename
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_state(path: Path) -> dict[str, Any]:
    """Read a state file and check its version.

    Args:
        path: the sweep_state.json of a sweep.

    Returns:
        The state dict.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the version is missing or not STATE_VERSION.
    """
    with open(path, encoding='utf-8') as f:
        state = json.load(f)

    version = state.get('version') if isinstance(state, dict) else None
    if version != STATE_VERSION:
        raise ValueError(f'{path}: state version must be {STATE_VERSION}, got {version!r}')
    return state
