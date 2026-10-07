# sweeper

sweeper runs a list of configs times repeats and commits every finished segment to disk, so a sweep that gets killed resumes from the last committed step instead of starting over. It knows nothing about the model, since a `Trial` you write does the training and sweeper owns the seeds, the checkpoints, the state file, and the order of work. Successive halving is available but off unless you pass it. The repo also holds `gpuq`, a single-file job queue for launching sweeps on the free GPUs of one workstation, which uses only the standard library.

## Running without an install

Python 3.10 or newer, torch 2.4 or newer, and numpy are required. wandb is an optional extra, and `sweeper.RunLogger` imports it only when a run starts, so a sweep with `logger=None` never needs it. Nothing has to be installed to use the package, and the test config sets `pythonpath = ["src"]`, so from the repo root

```
python -m pytest
```

finds `sweeper` and `gpuq` in `src/`. In your own scripts, put `src/` on `PYTHONPATH`. `pip install -e .` also works and adds the `gpuq` console script, and `pip install -e ".[wandb]"` adds wandb.

## Writing a Trial

A `Trial` is one (config, repeat) training run. `run_sweep` builds a fresh one from your factory for every segment and calls, in this order, `setup(params, seed, run_dir)`, `restore(ckpt)`, `train_until(epoch, logger, latest_ckpt)`, `save(ckpt)`, and `teardown()`. `restore` is skipped on a fresh start, and `train_until` is skipped when the score is already committed. The attribute `epoch` counts the epochs trained so far, so it is 0 after `setup` and must equal the target when `train_until` returns.

`train_until` writes `latest_ckpt` itself, every so often and at the end of the segment, to a tmp file followed by `os.replace`. It logs only through `logger.log` and never calls `logger.finish`, and it returns the segment's metric history as a dict from key to one value per epoch trained in that call. `restore` loads the checkpoint, sets `epoch` from it, and restores the global RNG states (`set_rng_states`) and the DataLoader generator state, so a resumed run draws what an unbroken one would. `teardown` can be called twice and after a `setup` that raised halfway. The sweeper has already seeded the global RNGs when it calls `setup`, and `run_dir` belongs to that (config, repeat) alone.

## The sweep file

The sweep file is JSON. Top-level keys are `version` (must be 1), `epochs`, and `configs`, all required, plus `n_repeats` (default 1), `base_params` (a dict, default `{}`), and `description`. Each entry of `configs` needs `name` and `params` (which may be `{}`) and takes `seeds` (a list of ints, one per repeat), `tags` (a list of strings), and `description`. A key outside these sets raises `ValueError`, as does a missing required key or a wrong version. `params` and `base_params` are opaque to sweeper, so the caller validates them.

```json
{
  "version": 1,
  "epochs": 40,
  "n_repeats": 3,
  "base_params": {"lr": 0.0002},
  "configs": [
    {"name": "small", "params": {"width": 64}},
    {"name": "large", "params": {"width": 256}, "tags": ["wide"]}
  ]
}
```

`resolve_params(defaults, base_params, cli, cli_keys, overrides)` layers the domain defaults, the sweep's `base_params`, the CLI values that were explicitly given, and the config's own `params`, in that order of increasing priority, so a config's `params` beat even an explicit flag. `explicit_flags(parser, argv)` returns the dests of the flags that appeared in argv, whatever their defaults are.

```python
args = parser.parse_args()
cli_keys = sweeper.explicit_flags(parser) & DEFAULTS.keys()
sweep = sweeper.load_sweep(Path(args.sweep))
configs = [
    sweeper.Config(
        name=c['name'],
        params=sweeper.resolve_params(
            DEFAULTS, sweep['base_params'], vars(args), cli_keys, c['params']
        ),
        seeds=tuple(c['seeds']) if 'seeds' in c else None,
        tags=tuple(c.get('tags', ())),
    )
    for c in sweep['configs']
]
```

A resume compares every config's params with the stored ones, so only keys the domain defaults know go into `cli_keys`, and run-control flags such as `--resume` stay out of the params.

## run_sweep, resume, and the output layout

```python
state = sweeper.run_sweep(
    configs,
    trial_factory,
    Path('runs/my_sweep'),
    sweep='my_sweep',
    epochs=sweep['epochs'],
    n_repeats=sweep['n_repeats'],
    base_seed=0,
    resume=False,
)
```

Every config runs repeat 0 before any config runs repeat 1. The returned dict is the final state, as written to `out_dir/sweep_state.json`. Config names must use letters, digits, `.`, `_`, and `-` with no leading `.`, since a name is also a directory name. `out_dir` holds:

```
sweep_state.json
sweep.lock                        held by the one run_sweep writing here
<config>/repeat<r>/latest.pth     written by the Trial while it trains
<config>/repeat<r>/rung<i>.pth    committed once per rung, a flat run has only rung0.pth
```

A second `run_sweep` on the same `out_dir` raises `RuntimeError` while the first holds `sweep.lock`. Without `resume`, an existing state file raises, and so does a `latest.pth` or `rung<i>.pth` left in a config's run dir, since a new sweep would restore it as its own. With `resume`, a missing state file raises. A resume raises, showing both values, when `sweep`, `epochs`, `n_repeats`, `halving`, or the `Score` settings differ from the state, when a config in the state is missing from the list, when a config's params differ from the stored ones (compared after a JSON round trip, so tuples equal lists, and the error names the differing keys), and when a seed differs from the stored one. A config added to the list is run as new in a flat sweep and raises under halving.

A segment commits its score to the state file first and the rung checkpoint second. After a kill at any point, a resume restores `latest.pth` when it exists, else the previous rung's checkpoint, and trains on from there. A kill that lands after `train_until` returned but before the score was written leaves a `latest.pth` at the target epoch with no stored score, so the history is gone, and sweeper warns and retrains that segment from the previous rung, and that retrain happens only when a `Score` was passed.

## Seeds

A repeat's seed is `config_seed(base_seed, sweep, config, repeat)`, a sha256 hash of those four values, so the same (sweep, config, repeat) gets the same seed in any run order. A config's explicit `seeds` list wins over the derived value for its repeats and must have `n_repeats` entries. Seeds are written to the state at launch, and a resume with a different `base_seed` or `seeds` raises. Before every segment sweeper calls `seed_everything(seed)`, which seeds python, numpy, and torch. That call also turns torch's deterministic kernels off, so a Trial that wants them turns them on in `setup`, which runs after the reseed.

## Halving

Halving is off unless `run_sweep` gets a `Halving(rung_epochs, survival_fraction, outlier_fence=None)` through its `halving` argument, and the sweep file has no key for it. It also needs a `Score`, or `run_sweep` raises.

Rung targets are every `rung_epochs` epochs up to `epochs`, plus `epochs` itself when it is not a multiple, so `rung_epochs=2` and `epochs=5` give `[2, 4, 5]`. A segment trains each repeat up to the next target, and the score is stored per repeat, and a config's rung score is the mean over its repeats.

After every rung except the last, the IQR prune and then the survival cut run over the configs not yet pruned, and the IQR prune runs only when `outlier_fence` is set. With the quartiles picked from the sorted scores without interpolation, the IQR prune drops scores above `Q3 + fence * IQR` for `'min'` and below `Q1 - fence * IQR` for `'max'`, and its fence does nothing with fewer than 4 finite scores. The survival cut then keeps the best `max(1, ceil(n * survival_fraction))` of the rest, ties broken by name, so `survival_fraction=1.0` with a fence gives an IQR-only prune. A NaN rung score warns and ranks after every finite one, and the IQR prune cuts it whenever a finite score exists. Every cut config gets `pruned_at` set to that rung, and the sweep stops early when one config is left.

`Score(keys, direction, top_fraction)` turns a segment's history into one number. Per key it averages the best `ceil(len * top_fraction)` epochs, and it averages those across keys with equal weight. `direction(key)` returns `'min'` when lower is better and `'max'` when higher is, and all keys of one score must agree.

## gpuq

gpuq launches shell one-liners from a queue file as GPUs free up, and runs under any python on the machine. Without an install, run it by path from a tmux pane:

```
python path/to/sweeper/src/gpuq.py validate queue.txt
python path/to/sweeper/src/gpuq.py run queue.txt
python path/to/sweeper/src/gpuq.py check
python path/to/sweeper/src/gpuq.py status
python path/to/sweeper/src/gpuq.py kill cuda:0
```

`validate` parses the live region and launches nothing, `run` is the watcher, `check` shows what the busy detector sees, `status` shows queued and running jobs, and `kill` stops a job. The `gpuq` console script exists only after a pip install, and takes the same subcommands.

`run` takes `--log-dir` (default `logs/gpuq`, relative to the working directory), `--floor-mib` (foreign CUDA contexts smaller than this are ignored, default 64), `--poll` (seconds between checks, default 20), `--settle` (seconds to let VRAM come back before an exclusive job, default 10), `--cwd` (working dir for jobs, default the current one), and `--sigint-grace` and `--sigterm-grace` (seconds before escalating to the next signal). `check` and `status` take `--log-dir` and `--floor-mib`. `kill` takes a target that is `cuda:N`, a GPU index, a job name, or `#jid`, plus `--log-dir` and `--floor-mib`, the same grace flags, `--yes` to skip the confirmation, `--requeue` to append the job's line to the end of the queue file, and `--resume` to add `--resume` to that appended line. The end of the file is below the barrier, so a requeued job runs only once you move its line above it.

The queue file is live, so anything above the barrier, a line of four or more `#`, is the queue, and anything below is a scratch area the watcher never touches. You can paste a line to add work, delete one to take it back, and reorder to reprioritize, all without restarting the watcher.

```
### RUNNING ON CUDA:0 ### python sweep.py --exp-tag a
python sweep.py --exp-tag c
[excl] python sweep.py --exp-tag e
[gpu=1] python sweep.py --exp-tag f
[name=sweep] python sweep.py --exp-tag g
[after=sweep] python analysis.py
# pause
#### gpuq barrier: nothing below this line runs ####
python sweep.py --exp-tag d
```

The watcher rewrites started jobs into `### RUNNING ON CUDA:N ###` lines, and finished jobs move to `<queue>.done.txt` with their exit code and runtime. A plain line gets a free GPU, and its `--device cuda:N` is retargeted to the GPU it got. `[excl]` waits for the whole box and runs verbatim, `[gpu=N]` pins a job to one GPU and runs verbatim, and `[name=...]` sets the log name. A job named by `[name=...]` or `--exp-tag` is the one `kill` and `[after=...]` can refer to, and without either it is named by its queue id. `[after=NAME]` holds a job until NAME exits 0, steps over it so the queue behind it keeps moving, and drops it if NAME exits nonzero. A `# pause` line anywhere above the barrier stops new launches while running jobs and bookkeeping carry on, and deleting the line resumes the queue.
