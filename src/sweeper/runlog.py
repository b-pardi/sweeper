"""Thin wandb wrapper that guards against backwards steps, wandb is only imported in init."""

import warnings
from collections.abc import Sequence
from typing import Any


class RunLogger:
    """Logs to one wandb project, one run at a time."""

    def __init__(self, project: str) -> None:
        """Holds the wandb project, nothing is started until init.

        Args:
            project: wandb project name every run goes into.
        """
        self.project = project
        self.run_id: str | None = None  # set by init, kept after finish so the caller can read it
        self._run: Any = None
        self._last_step: int | None = None
        self._drop_warned = False

    def init(
        self,
        name: str,
        *,
        tags: Sequence[str] = (),
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        """Start a wandb run, or pick an old one back up when run_id is given.

        Args:
            name: run name shown in wandb.
            tags: run tags.
            config: run config dict.
            run_id: id of an earlier run to resume. A fresh run when None.

        Returns:
            The wandb run id, so the caller can store it for a later resume.
        """
        import wandb

        kwargs: dict[str, Any] = {
            'project': self.project,
            'name': name,
            'tags': list(tags),
            'config': config,
        }
        if run_id is not None:
            kwargs['id'] = run_id
            kwargs['resume'] = 'allow'

        self._run = wandb.init(**kwargs)
        self.run_id = self._run.id

        # on resume run.step is the server's last step, so a client that went back gets dropped
        # a fresh run reports 0, which counts as no prior step
        server_step = self._run.step or 0
        self._last_step = server_step if server_step > 0 else None
        self._drop_warned = False
        return self.run_id

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics, dropping the call when step is below the last logged step.

        Does nothing before init or after finish. A stepless call is never dropped and never
        moves the last step.

        Args:
            metrics: values to log.
            step: wandb step, or None to let wandb pick.
        """
        if self._run is None:
            return

        # equal steps merge into one wandb row and several writers rely on it
        # only a lower step is a bad resume
        if step is not None and self._last_step is not None and step < self._last_step:
            if not self._drop_warned:
                msg = (
                    f'dropping log at step {step}, below the last logged step {self._last_step} '
                    '(further drops of this run are silent)'
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self._drop_warned = True
            return

        self._run.log(metrics, step=step)
        if step is not None:
            self._last_step = step

    def finish(self) -> None:
        """Finish the run and reset, safe to call when no run is open."""
        if self._run is None:
            return
        self._run.finish()
        self._run = None
        self._last_step = None
        self._drop_warned = False
