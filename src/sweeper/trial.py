"""The Trial protocol, the domain side that run_sweep drives one segment at a time."""

from pathlib import Path
from typing import Any, Protocol

from sweeper.runlog import RunLogger


class Trial(Protocol):
    """One (config, repeat) training run, built fresh by the trial factory for every segment.

    The sweeper picks every path and calls setup, restore, train_until, save, and teardown in that
    order. restore is skipped on a fresh start, train_until when the score is already committed.
    The reseed before each segment turns deterministic kernels off, so a Trial that wants them
    turns them on in setup.
    """

    epoch: int  # epochs trained so far, 0 after setup, set by restore from the checkpoint

    def setup(self, params: dict[str, Any], seed: int, run_dir: Path) -> None:
        """Build the model, data, and optimizer for params, and seed every generator from seed.

        The global RNGs are already seeded with seed. run_dir exists and belongs to this
        (config, repeat) alone, so anything the Trial writes goes there.

        Args:
            params: the config's params, opaque to the sweeper.
            seed: this repeat's seed.
            run_dir: <out_dir>/<config>/repeat<r>.
        """

    def restore(self, ckpt: Path) -> None:
        """Load a checkpoint from train_until or save, and set epoch from it.

        Also restores the global RNG states (sweeper.seeds.set_rng_states) and the DataLoader
        generator state, so a resumed run draws what an unbroken one would. A latest_ckpt brings
        back the segment's partial history too, a checkpoint from save starts the next segment
        with an empty one.

        Args:
            ckpt: latest_ckpt of an interrupted attempt at this segment, or the previous rung's
                checkpoint.
        """

    def train_until(
        self, epoch: int, logger: RunLogger | None, latest_ckpt: Path
    ) -> dict[str, list[float]]:
        """Train from self.epoch up to epoch, then return the segment's history since its start.

        The history is the partial one a restored latest_ckpt carried plus what this call trained,
        so with epoch == self.epoch it trains nothing and returns the stored one. Writes
        latest_ckpt with that history, every so often and at the end, to a tmp file then
        os.replace. Logs only through logger.log and never calls logger.finish.

        Args:
            epoch: target epoch, self.epoch must equal it on return.
            logger: the open run of this (config, repeat), or None to log nothing.
            latest_ckpt: where the resume checkpoint goes.

        Returns:
            key -> one value per epoch from the rung start up to epoch. A key logged only on eval
            epochs has fewer values.
        """

    def save(self, ckpt: Path) -> None:
        """Write the full training state to ckpt, the sweeper renames it after.

        ckpt carries no metric history, the segment's score is already committed.

        Args:
            ckpt: a tmp path next to the rung checkpoint.
        """

    def teardown(self) -> None:
        """Free what setup took. Safe to call twice and after a setup that raised halfway."""
