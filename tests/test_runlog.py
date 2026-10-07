import sys
import warnings
from contextlib import nullcontext

import pytest

from sweeper.config import Config
from sweeper.runlog import RunLogger
from sweeper.runner import run_sweep


@pytest.mark.parametrize(
    'server_step, run_id, steps, landed, n_warns',
    [
        (0, None, [3, 3, 3], [3, 3, 3], 0),  # equal steps all land
        (0, None, [5, 4, 3], [5], 1),  # second drop does not warn again
        (7, 'abc', [7, 8], [7, 8], 0),  # replaying the server's last step is fine
        (7, 'abc', [6, 7], [7], 1),  # below the server's last step is dropped
        (0, None, [5, None, 5], [5, None, 5], 0),  # stepless lands
        (7, None, [None, 6], [None], 1),  # stepless did not move the last step
    ],
)
def test_run_logger_same_step_rule(
    fake_wandb,
    fake_trial,
    monkeypatch,
    tmp_path,
    server_step: int,
    run_id: str | None,
    steps: list,
    landed: list,
    n_warns: int,
) -> None:
    """Only a lower step is dropped, the first drop warns once, and logger=None skips wandb."""
    fake_wandb.server_step = server_step
    logger = RunLogger('proj')

    out = logger.init('cfg-repeat0', tags=['a', 'b'], config={'lr': 0.1}, run_id=run_id)
    assert out == 'fake-run-id'
    assert logger.run_id == 'fake-run-id'

    kwargs = fake_wandb.init_kwargs[0]
    assert kwargs['project'] == 'proj'
    assert kwargs['name'] == 'cfg-repeat0'
    assert kwargs['tags'] == ['a', 'b']
    assert kwargs['config'] == {'lr': 0.1}
    if run_id is None:
        assert 'id' not in kwargs
        assert 'resume' not in kwargs
    else:
        assert kwargs['id'] == run_id
        assert kwargs['resume'] == 'allow'

    if n_warns:
        ctx = pytest.warns(RuntimeWarning, match='dropping')
    else:
        ctx = nullcontext()
    with ctx as rec, warnings.catch_warnings():
        warnings.simplefilter('always' if n_warns else 'error')
        for s in steps:
            logger.log({'m': s}, step=s)
    if n_warns:
        assert len(rec) == n_warns

    run = fake_wandb.runs[0]
    assert [s for s, _ in run.logged] == landed

    logger.finish()
    assert run.finished
    logger.log({'m': 0}, step=100)  # no run anymore, so a no-op
    assert len(run.logged) == len(landed)

    # a None module makes any import of wandb fail, so logger=None must never import it
    monkeypatch.setitem(sys.modules, 'wandb', None)
    state = run_sweep([Config('a', {})], fake_trial, tmp_path, sweep='s', epochs=1, base_seed=0)
    assert state['configs']['a']['repeats'][0]['run_id'] is None
