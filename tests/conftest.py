"""Shared fixtures for the sweeper tests."""

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest


class _FakeRun:
    """Stands in for a wandb run, records every log call."""

    def __init__(self, step: int) -> None:
        self.id = 'fake-run-id'
        self.step = step  # what the server says the last step is
        self.logged: list[tuple[int | None, dict[str, Any]]] = []
        self.finished = False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.logged.append((step, metrics))

    def finish(self) -> None:
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Put a fake wandb module in sys.modules, the real one is never imported."""
    mod = types.ModuleType('wandb')
    mod.server_step = 0  # the step the next init's run reports, like a resumed run
    mod.init_kwargs = []  # one dict per init call
    mod.runs = []  # one fake run per init call

    def init(**kwargs: Any) -> _FakeRun:
        run = _FakeRun(mod.server_step)
        mod.init_kwargs.append(kwargs)
        mod.runs.append(run)
        return run

    mod.init = init
    monkeypatch.setitem(sys.modules, 'wandb', mod)
    return mod


class FakeTrial:
    """Trial stand-in that keeps its checkpoints as small JSON files and records every call."""

    def __init__(self, kit: 'FakeTrials') -> None:
        self.kit = kit
        self.epoch = 0
        self.name = ''
        self.history: dict[str, list[float]] = {}  # the segment's so far, kept in latest

    def setup(self, params: dict[str, Any], seed: int, run_dir: Path) -> None:
        self.name = run_dir.parent.name  # the config name, from the sweeper's layout
        self.kit.calls.append(('setup', self.name, seed))

    def restore(self, ckpt: Path) -> None:
        data = json.loads(ckpt.read_text())
        self.epoch = data['epoch'] + self.kit.restore_shift
        # only latest has a history, a rung file starts the next segment empty
        self.history = data.get('history', {})
        self.kit.calls.append(('restore', ckpt.name, self.epoch))

    def train_until(self, epoch: int, logger: Any, latest_ckpt: Path) -> dict[str, list[float]]:
        self.kit.calls.append(('train_until', self.epoch, epoch))
        start = self.epoch
        crash_at = self.kit.crash.get(self.name)
        # a crash fires only in the segment whose epochs it falls between
        if crash_at is not None and not start < crash_at < epoch:
            crash_at = None
        self.epoch = epoch - self.kit.short_by if crash_at is None else crash_at

        script = self.kit.histories.get(self.name, {})
        new = {k: v[start : self.epoch] for k, v in script.items()}
        self.history = {k: self.history.get(k, []) + v for k, v in new.items()}

        # latest goes down before a crash too, like a periodic save would
        self._write(latest_ckpt)
        if crash_at is not None:
            raise RuntimeError(f'injected train_until failure at epoch {crash_at}')
        if 'train_until' in self.kit.fail and self.kit.kill_at in (None, epoch):
            raise RuntimeError('injected kill after train_until')
        return self.history

    def save(self, ckpt: Path) -> None:
        self.kit.calls.append(('save', ckpt.name))
        if 'save' in self.kit.fail:
            raise RuntimeError('injected save failure')
        ckpt.write_text(json.dumps({'epoch': self.epoch, 'payload': self.name}))

    def teardown(self) -> None:
        self.kit.calls.append(('teardown',))
        if 'teardown' in self.kit.fail:
            raise RuntimeError('injected teardown failure')

    def _write(self, path: Path) -> None:
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(
            json.dumps({'epoch': self.epoch, 'payload': self.name, 'history': self.history})
        )
        os.replace(tmp, path)


class FakeTrials:
    """trial_factory for run_sweep, every FakeTrial it builds shares these knobs and the log."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []  # every call of every trial, in order
        self.histories: dict[str, dict[str, list[float]]] = {}  # config -> key -> one per epoch
        self.fail: set[str] = set()  # 'save', 'teardown', or 'train_until' (after a full segment)
        self.crash: dict[str, int] = {}  # config -> epoch where train_until raises, mid segment
        self.short_by = 0  # train_until stops this many epochs before its target
        self.restore_shift = 0  # added to the epoch read back by restore
        self.kill_at: int | None = None  # with 'train_until' in fail, only this target's kill

    def __call__(self) -> FakeTrial:
        return FakeTrial(self)


@pytest.fixture
def fake_trial() -> FakeTrials:
    """A fresh FakeTrial factory, pass it to run_sweep as trial_factory."""
    return FakeTrials()
