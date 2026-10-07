#!/usr/bin/env python3
"""gpuq, a tiny GPU job queuer for a single workstation.

Paste shell one-liners into a queue file, run the watcher in a tmux pane, and it
launches them as GPUs free up. Jobs marked [excl] get the whole box to
themselves. Stdlib only, so it runs under any python on the machine.

The queue file is live. The watcher keeps it up to date while it runs, and you
can edit it at any time without restarting anything:

    ### RUNNING ON CUDA:0 ### python sweep.py --exp-tag a   <- started, on gpu 0
    ### RUNNING ON ALL GPUS ### [excl] python sweep.py ...  <- has the box to itself
    python sweep.py --exp-tag c                             <- queued, will run next
    #### gpuq barrier: nothing below this line runs ####
    python sweep.py --exp-tag d --half-writ                 <- staged, ignored

Anything above the barrier is the queue: paste a line to add work, delete one to
take it back, reorder to reprioritise. Anything below is a scratch area for
drafting long commands, and the watcher never touches it. Move a line above the
barrier (or drag the barrier down) when it is ready to go.

A barrier is any line of four or more '#'. Running markers use three, so they
never collide, and ordinary '#' comments stay comments.

Two more things you can do from the file while the watcher runs:

    # pause                                  <- stop starting new jobs
    [name=sweep] python sweep.py             <- give a job a name to depend on
    [after=sweep] python analysis.py         <- runs only after sweep exits 0

A '# pause' line anywhere above the barrier stops NEW launches. Reaping, marker
rewriting, the ledger and the jobs already running all carry on; delete the line
and the queue picks up again. It works from wherever it sits, which matters,
because the watcher renders running markers above the queued lines and so pushes
a pause typed at the top downward.

[after=NAME] holds a job until NAME has finished with exit 0. A job that is
waiting is stepped over rather than blocking the queue behind it, so an analysis
job parked behind a two-day sweep never idles the box. If its dependency exits
nonzero the job is dropped instead of running on bad input.

Finished jobs leave the queue and land in a sister file, queue.done.txt, with
their exit code and runtime.

Usage:
    python path/to/sweeper/src/gpuq.py validate queue.txt  # check the live region, launch nothing
    python path/to/sweeper/src/gpuq.py run queue.txt       # the watcher (foreground, for tmux)
    python path/to/sweeper/src/gpuq.py check               # what the busy detector sees right now
    python path/to/sweeper/src/gpuq.py status              # read the state from another pane
    python path/to/sweeper/src/gpuq.py kill cuda:0         # stop the job holding GPU 0
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_VERSION = 1

# A bare CUDA context costs a few hundred MiB, and a real job can idle down to
# roughly the same size, so an absolute threshold cannot tell them apart. This
# floor only drops trivial noise; DEFAULT_CONTEXT_SHARE does the real work.
DEFAULT_FLOOR_MIB = 64
# A stray context is always a process's smaller allocation, sitting next to a
# bigger one it is really training on. Anything at least this fraction of a
# process's own peak counts as real occupancy.
DEFAULT_CONTEXT_SHARE = 0.5

DEFAULT_POLL_SECS = 20.0
DEFAULT_SETTLE_SECS = 10.0
DEFAULT_SIGINT_GRACE = 20.0
DEFAULT_SIGTERM_GRACE = 15.0
DEFAULT_LOG_DIR = 'logs/gpuq'


class QueueParseError(ValueError):
    """Raised when a queue file line has a bad directive block."""


@dataclass
class Job:
    """One queued command plus whatever the directive block said about it.

    Attributes:
        jid: monotonically increasing queue id, used for log file names.
        name: short label for logs, from name= or --exp-tag or the jid.
        cmd: the command with the directive block stripped off, verbatim.
        exclusive: needs the whole box, no other job running anywhere.
        pin_gpu: physical GPU index to force, or None to let the queuer pick.
        raw_line: the original queue file line, for the audit log.
        lineno: 1-based line number in the queue file.
        after: names this job waits on. It only starts once every one of them
            has finished with exit 0.
    """

    jid: int
    name: str
    cmd: str
    exclusive: bool
    pin_gpu: int | None
    raw_line: str
    lineno: int
    after: list[str] = field(default_factory=list)


_EXP_TAG_RE = re.compile(r'--exp-tag[=\s]+(\S+)')
# needs an explicit = or space after --device so --device-map doesn't match
_DEVICE_RE = re.compile(r'(--device[=\s]+)cuda:(\d+)')


def parse_queue_line(line: str, lineno: int, jid: int) -> Job | None:
    """Turn one queue file line into a Job.

    Args:
        line: raw line from the queue file.
        lineno: 1-based line number, only used in error messages.
        jid: queue id to assign to this job.

    Returns:
        A Job, or None if the line was blank or a comment.

    Raises:
        QueueParseError: the [...] block is unclosed or holds an unknown key.
    """
    text = line.strip()
    if not text or text.startswith('#'):
        return None

    exclusive = False
    pin_gpu = None
    name = None
    after: list[str] = []

    if text.startswith('['):
        end = text.find(']')
        if end == -1:
            raise QueueParseError(
                f'line {lineno}: directive block opens with [ but never closes with ]: {text!r}'
            )
        body = text[1:end]
        text = text[end + 1 :].strip()
        for part in body.split(','):
            part = part.strip()
            if not part:
                continue
            key, sep, value = part.partition('=')
            key = key.strip()
            value = value.strip()
            if key == 'excl' and not sep:
                exclusive = True
            elif key == 'gpu' and sep:
                if not value.isdigit():
                    raise QueueParseError(
                        f'line {lineno}: gpu= wants an integer index, got {value!r}'
                    )
                pin_gpu = int(value)
            elif key == 'name' and sep:
                name = value
            elif key == 'after' and sep:
                if not value:
                    raise QueueParseError(
                        f'line {lineno}: after= wants the name of another job, got nothing'
                    )
                after.append(value)
            else:
                raise QueueParseError(
                    f'line {lineno}: unknown directive {part!r} '
                    '(known: excl, gpu=N, name=foo, after=foo)'
                )

    if name is None:
        match = _EXP_TAG_RE.search(text)
        name = match.group(1) if match else f'job{jid:04d}'

    return Job(
        jid=jid,
        name=name,
        cmd=text,
        exclusive=exclusive,
        pin_gpu=pin_gpu,
        raw_line=line.rstrip('\n'),
        lineno=lineno,
        after=after,
    )


def parse_queue_file(text: str, start_jid: int = 1, start_lineno: int = 1) -> list[Job]:
    """Parse a whole queue file body into Jobs, numbering from start_jid.

    Args:
        text: queue file contents, or just the un-ingested tail of one.
        start_jid: queue id to give the first real job.
        start_lineno: line number of the first line of `text` in the real file,
            so error messages and the audit log point at the right place.

    Returns:
        Jobs in file order. Comments and blanks are dropped but still count
        toward line numbers.

    Raises:
        QueueParseError: any line has a bad directive block.
    """
    jobs = []
    jid = start_jid
    for lineno, line in enumerate(text.splitlines(), start=start_lineno):
        job = parse_queue_line(line, lineno=lineno, jid=jid)
        if job is not None:
            jobs.append(job)
            jid += 1
    return jobs


def rewrite_device(cmd: str) -> tuple[str, int]:
    """Point every --device cuda:N at cuda:0.

    Auto-assigned jobs run under CUDA_VISIBLE_DEVICES=<physical>, which renumbers
    the claimed GPU to logical 0. A pasted --device cuda:1 would then blow up with
    an invalid device error, so we retarget it.

    Args:
        cmd: the command as written in the queue file.

    Returns:
        (rewritten command, number of occurrences actually changed). Already
        being cuda:0 counts as no change.
    """
    changed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        if match.group(2) != '0':
            changed += 1
        return f'{match.group(1)}cuda:0'

    return _DEVICE_RE.sub(repl, cmd), changed


@dataclass
class GpuState:
    """What nvidia-smi says about one physical GPU right now.

    Only foreign occupancy lives here. Jobs the queuer launched itself are
    tracked separately, because a freshly launched job has not allocated any
    VRAM yet and would otherwise look like an idle GPU.
    """

    index: int
    foreign_pids: list[int] = field(default_factory=list)
    foreign_mib: int = 0
    busy: bool = False
    ignored: list[tuple[int, int]] = field(default_factory=list)  # (pid, mib) strays


def _csv_rows(text: str) -> list[list[str]]:
    """Split nvidia-smi --format=csv output into rows, dropping the header."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return [[cell.strip() for cell in line.split(',')] for line in lines[1:]]


def _parse_mib(cell: str) -> int | None:
    """Pull an integer MiB out of a cell like '906 MiB'. None if unreadable."""
    match = re.match(r'(\d+)', cell)
    return int(match.group(1)) if match else None


def probe_gpus(
    uuid_csv: str,
    apps_csv: str,
    floor_mib: int = DEFAULT_FLOOR_MIB,
    context_share: float = DEFAULT_CONTEXT_SHARE,
) -> list[GpuState]:
    """Work out which GPUs are occupied by processes we did not launch.

    Telling a real job apart from a leftover CUDA context cannot be done on size
    alone: a bare context runs a few hundred MiB, and a real job between phases
    was measured at 388 MiB on this box. They overlap.

    What does separate them is shape. A stray context is always a process's
    SMALLER allocation, sitting beside the bigger one it is really working on.
    So a process counts as occupying a GPU when its allocation there is a decent
    fraction of its own peak across all GPUs. A lone job is its own peak and
    always counts; genuine multi-GPU training allocates comparably everywhere
    and counts on each.

    Args:
        uuid_csv: output of nvidia-smi --query-gpu=index,uuid --format=csv.
        apps_csv: output of nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory.
        floor_mib: drop anything below this outright, just to kill noise.
        context_share: fraction of a process's own peak that still counts as real.

    Returns:
        One GpuState per GPU, ordered by index.
    """
    index_by_uuid = {row[1]: int(row[0]) for row in _csv_rows(uuid_csv) if len(row) >= 2}
    states = {idx: GpuState(index=idx) for idx in sorted(index_by_uuid.values())}

    saw_unmappable = False
    seen = []  # (pid, gpu index, mib or None)
    for row in _csv_rows(apps_csv):
        if len(row) < 3:
            continue
        pid, uuid, mem_cell = int(row[0]), row[1], row[2]
        if uuid not in index_by_uuid:
            saw_unmappable = True
            continue
        seen.append((pid, index_by_uuid[uuid], _parse_mib(mem_cell)))

    peak: dict[int, int] = {}
    for pid, _, mib in seen:
        if mib is not None:
            peak[pid] = max(peak.get(pid, 0), mib)

    for pid, idx, mib in seen:
        state = states[idx]
        if mib is None:
            # cannot prove this gpu is idle, so treat it as taken
            state.foreign_pids.append(pid)
            state.busy = True
            continue
        if mib < floor_mib or mib < context_share * peak[pid]:
            state.ignored.append((pid, mib))
            continue
        state.foreign_pids.append(pid)
        state.foreign_mib += mib
        state.busy = True

    if saw_unmappable:
        # our picture of the box is incomplete; refuse to call anything free
        for state in states.values():
            state.busy = True

    return [states[idx] for idx in sorted(states)]


def parse_proc_stat(text: str) -> tuple[int, int]:
    """Pull (pgrp, starttime) out of the contents of /proc/<pid>/stat.

    Field 2 is the executable name in parentheses and may itself contain spaces
    and close-parens, so we split on the last ')' rather than on whitespace.

    Args:
        text: full contents of a /proc/<pid>/stat file.

    Returns:
        (process group id, start time in clock ticks since boot).
    """
    fields = _stat_fields(text)
    return int(fields[2]), int(fields[19])


def _stat_fields(text: str) -> list[str]:
    """Fields of /proc/<pid>/stat from field 3 on, so index i is field i+3."""
    return text.rpartition(')')[2].split()


def proc_starttime(pid: int) -> int | None:
    """Start time of a pid in clock ticks, or None if it is not running."""
    try:
        with open(f'/proc/{pid}/stat', encoding='utf-8') as handle:
            return parse_proc_stat(handle.read())[1]
    except (OSError, ValueError, IndexError):
        return None


def proc_alive(pid: int, starttime: int | None) -> bool:
    """Check a pid is alive AND is still the same process we started.

    Pids get recycled. Without the start time check, adopting a reused pid would
    have us babysitting, and possibly killing, a total stranger. Zombies count
    as gone: they keep their /proc entry and start time but have already exited
    and hold no GPU memory.
    """
    if starttime is None:
        return False
    try:
        with open(f'/proc/{pid}/stat', encoding='utf-8') as handle:
            fields = _stat_fields(handle.read())
        return fields[0] != 'Z' and int(fields[19]) == starttime
    except (OSError, ValueError, IndexError):
        return False


def group_members(pgid: int) -> list[int]:
    """Every live pid in a process group.

    Jobs are launched with setsid, so the whole tree shares the leader's pgid.
    This still finds the workers when the leader itself has exited.
    """
    members = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat', encoding='utf-8') as handle:
                fields = _stat_fields(handle.read())
            state, pgrp = fields[0], int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        # a zombie keeps its /proc entry and its pgid until someone reaps it,
        # but it holds no memory and is not running anything
        if state == 'Z':
            continue
        if pgrp == pgid:
            members.append(int(entry))
    return sorted(members)


@dataclass
class RunningJob:
    """A job the queuer launched (or reattached to) that is still alive.

    pid doubles as the process group id, since jobs are started with setsid.
    gpu is None for exclusive jobs, which hold the whole box.
    """

    jid: int
    name: str
    gpu: int | None
    pid: int
    starttime: int
    exclusive: bool
    log_path: str
    cmd_as_run: str
    started_at: float
    adopted: bool = False  # picked up from a previous watcher, so not waitable
    raw_line: str = ''  # the queue file line verbatim, directives and all


@dataclass
class Decision:
    """What the scheduler wants to do this tick."""

    action: str  # launch | wait | error
    reason: str
    job: Job | None = None
    gpu: int | None = None


_ASSIGNMENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')

# /bin/sh runs these itself, so which() will never find them on PATH
_SHELL_BUILTINS = frozenset(
    {
        '.',
        ':',
        '[',
        'alias',
        'bg',
        'break',
        'case',
        'cd',
        'command',
        'continue',
        'declare',
        'do',
        'done',
        'echo',
        'elif',
        'else',
        'esac',
        'eval',
        'exec',
        'exit',
        'export',
        'false',
        'fg',
        'fi',
        'for',
        'getopts',
        'hash',
        'if',
        'jobs',
        'local',
        'printf',
        'pwd',
        'read',
        'readonly',
        'return',
        'set',
        'shift',
        'source',
        'test',
        'then',
        'time',
        'times',
        'trap',
        'true',
        'type',
        'ulimit',
        'umask',
        'unalias',
        'unset',
        'until',
        'wait',
        'while',
        '{',
        '(',
    }
)


def launch_problem(cmd: str) -> str | None:
    """Say why a queue line could never run, or None if it looks runnable.

    The queue file is live, so a stray paste or a misdirected redirect can leave
    something that is not a command sitting in it. A pasted gpu-monitor screen
    row must never launch as a job, since it could just as easily be half of a
    real command.

    This is a structural check, not a validator: all it asks is whether the
    first word is something a shell could execute. Env assignments and shell
    builtins are fine, since neither ever appears on PATH.

    Args:
        cmd: the command with any [directive] block already stripped.

    Returns:
        A reason to refuse the line, or None if it looks like a command.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()  # unbalanced quote; the first word is still readable

    while tokens and _ASSIGNMENT_RE.match(tokens[0]):
        tokens = tokens[1:]  # FOO=bar prefixes are a legal way to start a command

    if not tokens:
        # nothing but assignments is a valid, if pointless, shell line
        return None if cmd.strip() else 'the line is empty'

    program = tokens[0]
    if program in _SHELL_BUILTINS or shutil.which(program):
        return None
    return (
        f'{program!r} is not a program: not on PATH, not an executable path, and '
        'not a shell builtin. Refusing to run this line (stray paste?)'
    )


def dependency_status(
    job: Job,
    running: list[RunningJob],
    pending: list[Job],
    finished: dict[str, int | None],
) -> tuple[str, str]:
    """Say whether a job's [after=...] dependencies are satisfied yet.

    Args:
        job: the job being considered.
        running: jobs we currently own.
        pending: the whole queue, so a dependency sitting BEHIND its dependent
            counts as "not yet" rather than as a typo.
        finished: name -> exit code for jobs that have already ended. None means
            the exit code is unknown, which happens when a job was adopted from a
            previous watcher and so was never ours to wait() on.

    Returns:
        (status, reason). "ready" to run; "wait" while a dependency is still
        going; "hold" when a dependency ended but we cannot prove it succeeded;
        "fail" when it can never be satisfied, so the job should be dropped
        rather than left in the queue forever.
    """
    for name in job.after:
        if name in finished:
            code = finished[name]
            if code == 0:
                continue
            if code is None:
                return 'hold', (
                    f'dependency {name!r} finished with an UNKNOWN exit code (adopted '
                    'from an earlier watcher, so we never saw it exit). Check its log, '
                    'then drop the [after=...] directive to release this job'
                )
            return 'fail', f'dependency {name!r} exited {code}, so this job would read bad input'
        if any(run.name == name for run in running):
            return 'wait', f'dependency {name!r} is still running'
        if any(other is not job and other.name == name for other in pending):
            return 'wait', f'dependency {name!r} is queued but has not started'
        return 'fail', (
            f'dependency {name!r} is not running, queued or finished. '
            'Nothing will ever satisfy it (typo?)'
        )
    return 'ready', ''


def decide(
    pending: list[Job],
    running: list[RunningJob],
    gpus: list[GpuState],
    now: float,
    last_exit_at: float,
    settle_secs: float,
    finished: dict[str, int | None] | None = None,
) -> Decision:
    """Work out the single next action, given a snapshot of the world.

    FIFO, except that a job waiting on an [after=...] dependency is stepped over
    instead of blocking everything behind it, otherwise an analysis job queued
    after a two-day sweep would idle the whole box for two days. Stepping over
    also means a dependency pasted BELOW its dependent still resolves. Nothing
    ever overtakes an exclusive job, because an exclusive job is only skipped
    when it declares a dependency of its own. Call this repeatedly, updating
    `running` after each launch, until it stops saying launch.

    Args:
        pending: queued jobs in order.
        running: jobs we currently own.
        gpus: foreign occupancy per GPU, from probe_gpus.
        now: current wall clock, seconds.
        last_exit_at: when one of our jobs last exited, seconds.
        settle_secs: how long to let VRAM come back before an exclusive start.
        finished: name -> exit code for jobs that have already ended.

    Returns:
        A Decision naming the job it refers to. "error" means that job can never
        run and should be failed out rather than blocking the queue forever.
    """
    if not pending:
        return Decision('wait', 'queue empty')

    job = None
    blocked = ''
    for candidate in pending:
        status, reason = dependency_status(candidate, running, pending, finished or {})
        if status == 'fail':
            return Decision('error', f'{candidate.name}: {reason}', job=candidate)
        if status != 'ready':
            blocked = blocked or f'{candidate.name}: {reason}'
            continue
        job = candidate
        break

    if job is None:
        return Decision('wait', blocked or 'queue empty')

    # checked before anything gpu-related: junk should not sit in the queue
    # waiting for a card it could never use
    problem = launch_problem(job.cmd)
    if problem:
        return Decision('error', f'{job.name}: {problem}', job=job)

    if any(r.exclusive for r in running):
        return Decision('wait', 'an exclusive job is running')

    if job.exclusive:
        if running:
            return Decision('wait', f'draining {len(running)} job(s) before exclusive {job.name}')
        busy = [g.index for g in gpus if g.busy]
        if busy:
            return Decision(
                'wait', f'exclusive {job.name} blocked by foreign activity on gpu {busy}'
            )
        remaining = settle_secs - (now - last_exit_at)
        if remaining > 0:
            return Decision('wait', f'settling before exclusive {job.name}, {remaining:.0f}s left')
        return Decision('launch', f'exclusive {job.name}, box is clear', job=job, gpu=None)

    by_index = {g.index: g for g in gpus}
    claimed = {r.gpu for r in running}

    if job.pin_gpu is not None:
        if job.pin_gpu not in by_index:
            return Decision(
                'error',
                f'job {job.name!r} pins gpu {job.pin_gpu}, which does not exist on this box',
                job=job,
            )
        if job.pin_gpu in claimed:
            return Decision('wait', f'pinned gpu {job.pin_gpu} held by one of our jobs')
        if by_index[job.pin_gpu].busy:
            return Decision('wait', f'pinned gpu {job.pin_gpu} busy with a foreign process')
        return Decision('launch', f'pinned to gpu {job.pin_gpu}', job=job, gpu=job.pin_gpu)

    for idx in sorted(by_index):
        if idx in claimed or by_index[idx].busy:
            continue
        return Decision('launch', f'gpu {idx} free', job=job, gpu=idx)

    return Decision('wait', 'no free gpu')


@dataclass
class State:
    """Everything the watcher needs to pick up where it left off.

    Rewritten atomically on every transition. The ledger next to it is an
    append-only audit trail; this file is the one that actually drives recovery.
    """

    queue_file: str = ''
    next_jid: int = 1
    pending: list[Job] = field(default_factory=list)
    running: list[RunningJob] = field(default_factory=list)
    last_exit_at: float = 0.0
    # name -> exit code, so an [after=name] job knows whether it may start. None
    # means the job ended but was not ours to wait() on, so the code is unknown.
    finished: dict[str, int | None] = field(default_factory=dict)
    paused: bool = False  # mirrors the queue file, so another pane can see it


def save_state(path: str | os.PathLike, state: State) -> None:
    """Write state to disk atomically.

    Writes a sibling temp file then renames it, so a watcher killed mid-write
    leaves the previous state intact rather than a truncated file.
    """
    path = Path(path)
    payload = {
        'version': STATE_VERSION,
        'queue_file': state.queue_file,
        'next_jid': state.next_jid,
        'last_exit_at': state.last_exit_at,
        'finished': state.finished,
        'paused': state.paused,
        'pending': [asdict(job) for job in state.pending],
        'running': [asdict(run) for run in state.running],
    }
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def load_state(path: str | os.PathLike) -> State | None:
    """Read state back, or None if the watcher has never run here."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    return State(
        queue_file=data.get('queue_file', ''),
        next_jid=data.get('next_jid', 1),
        last_exit_at=data.get('last_exit_at', 0.0),
        finished=data.get('finished', {}),
        paused=data.get('paused', False),
        pending=[Job(**item) for item in data.get('pending', [])],
        running=[RunningJob(**item) for item in data.get('running', [])],
    )


BARRIER_RE = re.compile(r'^\s*#{4,}')
RUNNING_MARKER_RE = re.compile(r'^\s*###\s+RUNNING\s+ON\s+')
# one to three hashes, so it can never be mistaken for a barrier. \b keeps a
# comment like "# paused this yesterday" from stopping the queue.
PAUSE_RE = re.compile(r'^\s*#{1,3}\s*(?:gpuq\s*:?\s*)?pause\b', re.IGNORECASE)
CANONICAL_BARRIER = (
    '#### gpuq barrier: nothing below this line runs. paste here, move up when ready ####'
)


@dataclass
class QueueDocument:
    """A queue file split into the parts the watcher may and may not touch.

    Above the barrier the watcher is in charge: it rewrites running markers and
    drops lines as they get launched. Below it is yours to scribble in.
    """

    content: list[str] = field(default_factory=list)  # live region, verbatim
    staging: list[str] = field(default_factory=list)  # below the barrier, untouched
    barrier: str | None = None
    running_lines: list[str] = field(default_factory=list)  # markers we wrote before
    paused: bool = False  # a '# pause' line in the live region


def parse_queue_document(text: str) -> QueueDocument:
    """Split a queue file at the first barrier line.

    A barrier is any line starting with four or more '#'. Running markers use
    exactly three, so the two can never be confused, and ordinary one- or
    two-hash comments stay ordinary comments.

    A '# pause' comment anywhere in the live region sets doc.paused. Position is
    deliberately not part of it: the watcher renders running markers above the
    queued lines, so a pause typed at the very top gets pushed down the moment a
    job starts, and a top-anchored flag would quietly defeat itself.
    """
    doc = QueueDocument()
    past_barrier = False
    for raw in text.splitlines():
        if past_barrier:
            doc.staging.append(raw)
        elif BARRIER_RE.match(raw):
            doc.barrier = raw
            past_barrier = True
        elif RUNNING_MARKER_RE.match(raw):
            doc.running_lines.append(raw)
        else:
            if PAUSE_RE.match(raw):
                doc.paused = True
            doc.content.append(raw)
    return doc


def running_marker(run: RunningJob) -> str:
    """The '### RUNNING ON ... ###' line for a live job."""
    where = 'ALL GPUS' if run.gpu is None else f'CUDA:{run.gpu}'
    return f'### RUNNING ON {where} ### {run.raw_line or run.cmd_as_run}'


def render_queue_document(running: list[RunningJob], content: list[str], staging: list[str]) -> str:
    """Rebuild the queue file: running jobs on top, then queued work, then staging."""
    lines = [running_marker(run) for run in running]
    lines.extend(content)
    lines.append(CANONICAL_BARRIER)
    lines.extend(staging)
    return '\n'.join(lines) + '\n'


def done_entry(run: RunningJob, exit_code: int | None, duration: float) -> str:
    """One record for the done file: what ran, when it ended, and how it went."""
    where = 'all-gpus' if run.gpu is None else f'cuda:{run.gpu}'
    shown = 'unknown' if exit_code is None else exit_code
    stamp = time.strftime('%Y-%m-%d %H:%M')
    return (
        f'### DONE {stamp} exit={shown} {_age(duration)} {where} ###\n'
        f'  {run.raw_line or run.cmd_as_run}\n'
    )


@dataclass
class ReattachResult:
    """What a previous watcher left behind."""

    alive: list[RunningJob] = field(default_factory=list)
    dead: list[RunningJob] = field(default_factory=list)
    orphans: list[tuple[RunningJob, list[int]]] = field(default_factory=list)


def reattach(running: list[RunningJob]) -> ReattachResult:
    """Sort jobs from a previous watcher into still-alive and finished.

    A job is only adopted when its pid is alive AND its start time still
    matches, so a recycled pid can never be mistaken for our process.

    Leftover process groups whose leader has died are deliberately NOT adopted:
    we cannot prove a reused pgid is ours, and signalling a stranger later would
    be far worse than the alternative. They get reported instead, and if they
    are really holding VRAM the foreign-occupancy check keeps the GPU blocked.

    Args:
        running: jobs restored from state.json.

    Returns:
        ReattachResult with alive (adopted), dead, and orphans as
        (job, leftover pids) pairs.
    """
    result = ReattachResult()
    for job in running:
        if proc_alive(job.pid, job.starttime):
            job.adopted = True
            result.alive.append(job)
            continue
        leftovers = group_members(job.pid)
        if leftovers:
            result.orphans.append((job, leftovers))
        result.dead.append(job)
    return result


def kill_group(
    pgid: int,
    sigint_grace: float = 20.0,
    sigterm_grace: float = 15.0,
    poll: float = 0.2,
) -> str:
    """Stop a whole job tree, escalating politely.

    SIGINT first so python raises KeyboardInterrupt and any finally blocks and
    W&B teardown get to run, then SIGTERM, then SIGKILL for anything wedged in a
    CUDA call. Signals go to the process group, so dataloader workers go too.

    Args:
        pgid: the job's process group id (same as its launch pid).
        sigint_grace: seconds to wait after SIGINT before escalating.
        sigterm_grace: seconds to wait after SIGTERM before SIGKILL.
        poll: how often to re-check whether the group is gone.

    Returns:
        Name of the signal that finished it, "already gone", or "still alive"
        if even SIGKILL did not clear it.
    """
    if not group_members(pgid):
        return 'already gone'

    for name, sig, grace in (
        ('SIGINT', signal.SIGINT, sigint_grace),
        ('SIGTERM', signal.SIGTERM, sigterm_grace),
        ('SIGKILL', signal.SIGKILL, 5.0),
    ):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return 'already gone'
        deadline = time.monotonic() + grace
        while True:
            if not group_members(pgid):
                return name
            if time.monotonic() >= deadline:
                break
            time.sleep(poll)

    return 'still alive'


def resolve_kill_target(
    target: str,
    running: list[RunningJob],
    gpus: list[GpuState],
) -> tuple[RunningJob | None, str | None]:
    """Find the job a kill request refers to.

    Accepts cuda:N, gpuN, a bare index, a job name, or #jid.

    Returns:
        (job, None) on a hit, or (None, message) explaining why not. Processes
        gpuq did not launch are never resolvable: the queuer only ever signals
        its own children.
    """
    text = target.strip()
    low = text.lower()
    for prefix in ('cuda:', 'gpu:', 'gpu'):
        if low.startswith(prefix):
            low = low[len(prefix) :]
            break
    index = int(low) if low.isdigit() else None

    if index is None:
        wanted = text.lstrip('#')
        for job in running:
            if job.name == text or str(job.jid) == wanted:
                return job, None
        return None, (
            f'no running gpuq job matches {target!r}; '
            'expected cuda:N, a gpu index, a job name, or #jid'
        )

    for job in running:
        if job.exclusive or job.gpu == index:
            return job, None

    state = {gpu.index: gpu for gpu in gpus}.get(index)
    if state is None:
        return None, f'gpu {index} does not exist on this box'
    if state.foreign_pids:
        return None, (
            f'gpu {index} is held by pid(s) {state.foreign_pids}, which gpuq '
            'did not launch. refusing to signal it, stop it yourself'
        )
    return None, f'no gpuq job is running on gpu {index}'


_UNSAFE_NAME_RE = re.compile(r'[^A-Za-z0-9._-]+')


class WatcherLockError(RuntimeError):
    """Another live watcher already owns this log directory."""


def _run_smi(args: list[str], timeout: float) -> str:
    """Run nvidia-smi and hand back stdout, or raise with something readable."""
    try:
        done = subprocess.run(
            ['nvidia-smi', *args], capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise RuntimeError('nvidia-smi not found on PATH') from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'nvidia-smi timed out after {timeout}s') from exc
    if done.returncode != 0:
        raise RuntimeError(f'nvidia-smi failed ({done.returncode}): {done.stderr.strip()}')
    return done.stdout


def nvidia_smi_probe(floor_mib: int = DEFAULT_FLOOR_MIB, timeout: float = 15.0) -> list[GpuState]:
    """Ask the driver what is currently running on each GPU."""
    uuid_csv = _run_smi(['--query-gpu=index,uuid', '--format=csv'], timeout)
    apps_csv = _run_smi(['--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv'], timeout)
    return probe_gpus(uuid_csv, apps_csv, floor_mib)


def _safe_name(name: str) -> str:
    """Squash a job name into something safe for a filename."""
    return _UNSAFE_NAME_RE.sub('-', name).strip('-') or 'job'


class Watcher:
    """The scheduling loop: reap, ingest, launch, repeat.

    Owns a log directory holding one log per job plus state.json (recovery),
    ledger.jsonl (audit) and watcher.lock. Everything that needs a GPU view goes
    through gpu_probe, which tests replace with a fake.
    """

    def __init__(
        self,
        queue_file: str | os.PathLike,
        log_dir: str | os.PathLike,
        gpu_probe=None,
        poll_secs: float = DEFAULT_POLL_SECS,
        settle_secs: float = DEFAULT_SETTLE_SECS,
        floor_mib: int = DEFAULT_FLOOR_MIB,
        cwd: str | None = None,
        sigint_grace: float = DEFAULT_SIGINT_GRACE,
        sigterm_grace: float = DEFAULT_SIGTERM_GRACE,
    ) -> None:
        self.queue_file = Path(queue_file)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_probe = gpu_probe or (lambda: nvidia_smi_probe(floor_mib))
        self.poll_secs = poll_secs
        self.settle_secs = settle_secs
        self.floor_mib = floor_mib
        self.cwd = cwd or os.getcwd()
        self.sigint_grace = sigint_grace
        self.sigterm_grace = sigterm_grace

        self.state_path = self.log_dir / 'state.json'
        self.ledger_path = self.log_dir / 'ledger.jsonl'
        self.lock_path = self.log_dir / 'watcher.lock'
        # sister file next to the queue, e.g. queue.txt -> queue.done.txt
        self.done_path = self.queue_file.with_name(
            f'{self.queue_file.stem}.done{self.queue_file.suffix or ".txt"}'
        )

        self.state = load_state(self.state_path) or State(queue_file=str(self.queue_file))
        self.last_warning: str | None = None
        self.last_reason = ''
        self.draining = False
        self.detached = False
        self.paused = self.state.paused
        self._held_warned: set[str] = set()  # so a held job warns once, not every tick

        self._procs: dict[int, object] = {}  # jid -> Popen, only for this session
        self._detached_procs: list[object] = []
        self._holds_lock = False
        self._doc: QueueDocument | None = None
        self._content: list[str] = []
        self._raw_text = ''
        # deferred to the first tick on purpose: constructing a Watcher must not
        # write anything, so one that gets locked out never touches the ledger
        self._reattached = False

    # ------------------------------------------------------------ bookkeeping

    def _ledger(self, event: str, **fields) -> None:
        record = {'ts': round(time.time(), 3), 'event': event, **fields}
        with open(self.ledger_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')

    def _warn(self, message: str) -> None:
        self.last_warning = message
        print(f'[gpuq] WARNING: {message}', flush=True)

    def _say(self, message: str) -> None:
        print(f'[gpuq] {message}', flush=True)

    # -------------------------------------------------------------- lifecycle

    def acquire_lock(self) -> None:
        """Claim this log dir, or refuse if a live watcher already holds it.

        Raises:
            WatcherLockError: another watcher process is alive and owns it.
        """
        if self.lock_path.exists():
            try:
                held = json.loads(self.lock_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                held = {}
            pid, starttime = held.get('pid'), held.get('starttime')
            if pid and proc_alive(pid, starttime):
                raise WatcherLockError(
                    f'another gpuq watcher (pid {pid}) is already running on {self.log_dir}'
                )
            self._say(f'clearing a stale lock from pid {pid}')
        me = os.getpid()
        self.lock_path.write_text(
            json.dumps({'pid': me, 'starttime': proc_starttime(me)}), encoding='utf-8'
        )
        self._holds_lock = True

    def release_lock(self) -> None:
        if self._holds_lock:
            try:
                self.lock_path.unlink()
            except OSError:
                pass
            self._holds_lock = False

    def begin_drain(self) -> None:
        """Stop launching new work; let what is running finish."""
        self.draining = True

    def detach(self) -> None:
        """Walk away without signalling anything. Jobs keep running."""
        self.detached = True
        self._detached_procs.extend(self._procs.values())
        self._procs.clear()

    def on_interrupt(self) -> None:
        """Ctrl-C: first one drains, second one detaches."""
        if not self.draining:
            self.begin_drain()
            self._say(
                f'draining. no new launches; waiting on {len(self.state.running)} job(s). '
                'Ctrl-C again to leave them running and exit now.'
            )
        else:
            self.detach()

    def shutdown_hard(self) -> None:
        """Kill everything we launched. For tests and emergencies only."""
        for run in list(self.state.running):
            try:
                kill_group(run.pid, sigint_grace=0.2, sigterm_grace=0.2, poll=0.05)
            except (OSError, ProcessLookupError):
                pass
        for proc in list(self._procs.values()) + self._detached_procs:
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        self._procs.clear()
        self._detached_procs.clear()
        self.release_lock()

    # ---------------------------------------------------------------- reattach

    def _reattach(self) -> None:
        self._reattached = True
        if not self.state.running:
            return
        result = reattach(self.state.running)
        for job, leftovers in result.orphans:
            self._warn(
                f'job #{job.jid} ({job.name}): leader pid {job.pid} is gone but pids '
                f'{leftovers} still share its process group. Not adopting them, since a '
                'reused group id cannot be told apart from the real one. Check by hand.'
            )
        for job in result.dead:
            self._finish(job, exit_code=None, detected_by='reattach')
        for job in result.alive:
            self._say(
                f'reattached #{job.jid} {job.name} (gpu {job.gpu}, pid {job.pid}) -> {job.log_path}'
            )
            self._ledger('reattached', jid=job.jid, name=job.name, pid=job.pid, gpu=job.gpu)
        self.state.running = result.alive
        save_state(self.state_path, self.state)

    # ------------------------------------------------------------------- ticks

    def _finish(self, run: RunningJob, exit_code: int | None, detected_by: str = 'wait') -> None:
        duration = max(0.0, time.time() - run.started_at)
        shown = 'unknown' if exit_code is None else exit_code
        try:
            with open(run.log_path, 'a', encoding='utf-8') as handle:
                handle.write(
                    f'\n--- gpuq: exit_code={shown} duration={duration:.1f}s '
                    f'detected_by={detected_by} ---\n'
                )
        except OSError:
            pass
        self._say(f'#{run.jid} {run.name} finished, exit_code={shown} after {duration:.0f}s')
        self._ledger(
            'finished',
            jid=run.jid,
            name=run.name,
            gpu=run.gpu,
            pid=run.pid,
            exit_code=exit_code,
            duration_s=round(duration, 1),
            detected_by=detected_by,
        )
        self._record_done(run, exit_code, duration)
        self.state.finished[run.name] = exit_code
        self.state.last_exit_at = time.time()
        self._procs.pop(run.jid, None)

    def _reap(self) -> None:
        still = []
        for run in self.state.running:
            proc = self._procs.get(run.jid)
            if proc is not None:
                code = proc.poll()
                if code is None:
                    still.append(run)
                    continue
                self._finish(run, code)
            else:
                # adopted from a previous watcher: not our child, so no exit code
                if proc_alive(run.pid, run.starttime):
                    still.append(run)
                    continue
                self._finish(run, None, detected_by='orphan-poll')
        self.state.running = still

    def _sync_queue(self) -> None:
        """Re-derive the pending queue from the live region of the queue file.

        The file is the queue: whatever plain lines sit above the barrier are
        what is waiting, so pasting adds work and deleting a line takes it back.
        Running jobs appear only as markers, which we regenerate from state, so a
        stale editor save that turns a marker back into a plain line is caught
        here instead of launching the job a second time.
        """
        try:
            text = self.queue_file.read_text(encoding='utf-8')
        except OSError as exc:
            self._warn(f'cannot read queue file {self.queue_file}: {exc}')
            self._doc = None
            return

        doc = parse_queue_document(text)
        self._doc = doc
        self._raw_text = text

        if doc.paused != self.paused:
            self.paused = doc.paused
            self._say(
                'paused: no new launches until the pause line goes away'
                if doc.paused
                else 'resumed: launching again'
            )
            self._ledger('paused' if doc.paused else 'resumed')

        # Only suspect a wiped marker when markers have actually gone missing.
        # Without this budget, a queued line that merely looks like a running one
        # would be swallowed as a phantom clobber and never run.
        missing = max(0, len(self.state.running) - len(doc.running_lines))
        unclaimed = collections.Counter(
            (run.raw_line or run.cmd_as_run).strip() for run in self.state.running
        )
        jobs: list[Job] = []
        kept: list[str] = []
        clobbered = 0

        for lineno, raw in enumerate(doc.content, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith('#'):
                kept.append(raw)
                continue
            if missing > 0 and unclaimed.get(stripped, 0) > 0:
                unclaimed[stripped] -= 1
                missing -= 1
                clobbered += 1
                continue
            try:
                job = parse_queue_line(raw, lineno=lineno, jid=len(jobs) + 1)
            except QueueParseError as exc:
                self._warn(str(exc))
                kept.append(raw)
                continue
            if job is None:
                kept.append(raw)
                continue
            job.raw_line = raw
            jobs.append(job)
            kept.append(raw)

        if clobbered:
            self._warn(
                f'{clobbered} queue line(s) match a job that is already running '
                '(stale editor save?). Re-asserting the RUNNING marker rather than '
                'launching it again.'
            )

        self.state.pending = jobs
        self._content = kept

    def _warn_about_held_jobs(self) -> None:
        """Shout about jobs whose dependency ended with an unknown exit code.

        These sit in the queue indefinitely on purpose: nothing wrong runs, and
        the skip in decide() keeps the rest of the queue moving. But a silently
        stuck job is its own kind of bug, so say so once each.
        """
        for job in self.state.pending:
            status, reason = dependency_status(
                job, self.state.running, self.state.pending, self.state.finished
            )
            message = f'{job.name} is held: {reason}'
            if status == 'hold' and message not in self._held_warned:
                self._held_warned.add(message)
                self._warn(message)

    def _rewrite_queue(self) -> None:
        """Write the queue file back, but only if it would actually change."""
        if self._doc is None:
            return
        text = render_queue_document(self.state.running, self._content, self._doc.staging)
        if text == self._raw_text:
            return
        tmp = self.queue_file.with_name(self.queue_file.name + '.gpuq.tmp')
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, self.queue_file)
        self._raw_text = text

    def _drop_from_queue(self, job: Job) -> None:
        """Take a launched or failed job's line out of the live region."""
        try:
            self._content.remove(job.raw_line)
        except ValueError:
            pass

    def _record_done(self, run: RunningJob, exit_code: int | None, duration: float) -> None:
        with open(self.done_path, 'a', encoding='utf-8') as handle:
            handle.write(done_entry(run, exit_code, duration) + '\n')

    def _launch(self, job: Job, gpu: int | None) -> None:
        # jids are handed out at launch, not at parse time: the pending list is
        # re-derived from the file every tick, so parse-time numbering would churn
        job.jid = self.state.next_jid
        self.state.next_jid += 1
        if re.fullmatch(r'job\d+', job.name):
            job.name = f'job{job.jid:04d}'

        log_path = self.log_dir / f'{job.jid:04d}-{_safe_name(job.name)}.log'
        env = dict(os.environ)
        # whatever the watcher's own shell had is never what a job should inherit
        env.pop('CUDA_VISIBLE_DEVICES', None)

        cmd, rewrites = job.cmd, 0
        if gpu is not None and job.pin_gpu is None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            cmd, rewrites = rewrite_device(job.cmd)
        cvd = env.get('CUDA_VISIBLE_DEVICES', '')

        header = [
            f'=== gpuq job #{job.jid:04d} {job.name} ===',
            f'started    {time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'gpu        {"ALL (exclusive)" if gpu is None else gpu}  CUDA_VISIBLE_DEVICES={cvd!r}',
            f'cwd        {self.cwd}',
            f'queue line {job.lineno}',
            f'command    {cmd}',
        ]
        if rewrites:
            header.append(f'original   {job.cmd}')
            header.append(
                f'note       rewrote {rewrites} --device flag(s) to cuda:0, because '
                f'CUDA_VISIBLE_DEVICES={cvd} renumbers physical gpu {gpu} to logical 0'
            )
        header.append('-' * 60)
        with open(log_path, 'a', encoding='utf-8') as handle:
            handle.write('\n'.join(header) + '\n')

        log = open(log_path, 'a', encoding='utf-8', buffering=1)
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self.cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group: Ctrl-C here never reaches it
            )
        finally:
            log.close()

        run = RunningJob(
            jid=job.jid,
            name=job.name,
            gpu=gpu,
            pid=proc.pid,
            starttime=proc_starttime(proc.pid) or 0,
            exclusive=job.exclusive,
            log_path=str(log_path),
            cmd_as_run=cmd,
            started_at=time.time(),
            raw_line=job.raw_line,
        )
        self._procs[job.jid] = proc
        self.state.running.append(run)
        self._drop_from_queue(job)
        self._say(
            f'started #{job.jid} {job.name} on '
            f'{"ALL gpus" if gpu is None else f"gpu {gpu}"} (pid {proc.pid}) -> {log_path}'
        )
        self._ledger(
            'started',
            jid=job.jid,
            name=job.name,
            gpu=gpu,
            pid=proc.pid,
            starttime=run.starttime,
            cmd_as_run=cmd,
            log=str(log_path),
            cuda_visible_devices=cvd,
            device_rewrites=rewrites,
        )

    def tick(self) -> None:
        """One scheduling pass: reap finished jobs, take in new ones, launch."""
        if not self._reattached:
            self._reattach()
        self._reap()
        self._sync_queue()
        self._warn_about_held_jobs()

        if self.paused:
            self.last_reason = (
                f"paused by a '# pause' line in {self.queue_file.name}; "
                f'{len(self.state.running)} job(s) still running'
            )

        if not self.draining and not self.paused:
            try:
                gpus = self.gpu_probe()
            except Exception as exc:
                # no view of the gpus means no safe launch decision
                self._warn(f'gpu probe failed, holding the queue: {exc}')
                gpus = None
            if gpus is not None:
                while True:
                    decision = decide(
                        self.state.pending,
                        self.state.running,
                        gpus,
                        time.time(),
                        self.state.last_exit_at,
                        self.settle_secs,
                        self.state.finished,
                    )
                    self.last_reason = decision.reason
                    if decision.action == 'launch':
                        # not pop(0): a job blocked on a dependency gets stepped
                        # over, so the chosen job is not always the head
                        self.state.pending.remove(decision.job)
                        self._launch(decision.job, decision.gpu)
                        continue
                    if decision.action == 'error':
                        job = decision.job
                        self.state.pending.remove(job)
                        self._warn(decision.reason)
                        self._ledger('error', name=job.name, reason=decision.reason)
                        # take it out of the file too, or it errors again every tick
                        self._drop_from_queue(job)
                        with open(self.done_path, 'a', encoding='utf-8') as handle:
                            handle.write(
                                f'### ERROR {time.strftime("%Y-%m-%d %H:%M")} '
                                f'{decision.reason} ###\n  {job.raw_line}\n\n'
                            )
                        continue
                    break

        self._rewrite_queue()
        self.state.paused = self.paused
        save_state(self.state_path, self.state)

    def run(self) -> None:
        """Watch until Ctrl-C. Jobs always outlive the watcher."""
        self.acquire_lock()
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: self.on_interrupt())
        self._say(
            f'watching {self.queue_file} every {self.poll_secs:.0f}s '
            f'(settle {self.settle_secs:.0f}s, foreign floor {self.floor_mib} MiB)'
        )
        try:
            last_note = None
            while True:
                self.tick()
                if self.detached:
                    break
                if self.draining and not self.state.running:
                    break
                if self.last_reason != last_note:
                    self._say(
                        f'pending={len(self.state.pending)} running={len(self.state.running)}'
                        f' | {self.last_reason}'
                    )
                    last_note = self.last_reason
                # sleep in slices so Ctrl-C feels immediate
                waited = 0.0
                while waited < self.poll_secs and not self.detached:
                    time.sleep(min(0.5, self.poll_secs - waited))
                    waited += 0.5
        finally:
            signal.signal(signal.SIGINT, previous)
            self.release_lock()

        if self.state.running:
            self._say('exiting, these jobs keep running:')
            for run in self.state.running:
                self._say(f'  #{run.jid} {run.name} gpu={run.gpu} pid={run.pid} {run.log_path}')
            self._say('start the watcher again to reattach to them.')


def requeue_line(name: str, cmd: str, add_resume: bool = False) -> str:
    """Build a queue file line that re-runs a job.

    Args:
        name: job name, so the requeued job keeps its log naming.
        cmd: the command as it was actually run.
        add_resume: append --resume if it is not already there. Opt in, because
            it is only right for scripts that checkpoint.
    """
    if add_resume and '--resume' not in cmd:
        cmd = f'{cmd} --resume'
    return f'[name={name}] {cmd}'


def _age(seconds: float) -> str:
    if seconds < 90:
        return f'{seconds:.0f}s'
    if seconds < 5400:
        return f'{seconds / 60:.0f}m'
    return f'{seconds / 3600:.1f}h'


def ensure_queue_file(path: str | os.PathLike) -> bool:
    """Create an empty queue file with a barrier if it is not there yet.

    Returns:
        True if it created one, False if the file already existed. Never
        overwrites.
    """
    path = Path(path)
    if path.exists():
        return False
    path.write_text(
        '# gpuq queue. lines above the barrier run; below it is yours to draft in.\n'
        '#   [excl] cmd       whole box to itself\n'
        '#   [gpu=1] cmd      pinned, run verbatim\n'
        '#   [name=foo] cmd   explicit log name\n'
        '#   [after=foo] cmd  waits for job foo to finish with exit 0\n'
        "#   a line reading '#pause' stops new launches; delete it to resume\n"
        f'{CANONICAL_BARRIER}\n',
        encoding='utf-8',
    )
    return True


def cmd_validate(args) -> int:
    """Parse the live region of the queue file and print what would happen."""
    path = Path(args.queue_file)
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        print(f'cannot read queue file {path}: {exc}', file=sys.stderr)
        return 2

    doc = parse_queue_document(text)
    try:
        jobs = parse_queue_file('\n'.join(doc.content))
    except QueueParseError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if doc.paused:
        print("PAUSED: a '# pause' line is in the live region, so the watcher will")
        print('        not start anything new. Delete that line to resume.\n')

    if doc.running_lines:
        print(f'{len(doc.running_lines)} job(s) already running:')
        for line in doc.running_lines:
            print(f'  {line}')
        print()

    if not jobs:
        print(f'{path} has no queued jobs above the barrier')
        if doc.staging:
            print(f'({len(doc.staging)} line(s) staged below the barrier, not checked)')
        return 0

    print(f'{len(jobs)} job(s) in {path}:')
    for job in jobs:
        flags = []
        if job.exclusive:
            flags.append('excl')
        if job.pin_gpu is not None:
            flags.append(f'gpu={job.pin_gpu}')
        flags.extend(f'after={name}' for name in job.after)
        tag = f'  [{", ".join(flags)}]' if flags else ''
        print(f'\n  #{job.jid:04d} {job.name}{tag}   (line {job.lineno})')
        print(f'    {job.cmd}')
        for name in job.after:
            known = any(other.name == name for other in jobs)
            note = '' if known else '  <- NOT IN THIS QUEUE, would be failed out'
            print(f'    -> waits for {name} to finish with exit 0{note}')
        problem = launch_problem(job.cmd)
        if problem:
            print(f'    -> WILL NOT RUN: {problem}')
            continue
        if job.exclusive:
            print('    -> whole box to itself, command run verbatim')
        elif job.pin_gpu is not None:
            print(f'    -> pinned to gpu {job.pin_gpu}, command run verbatim')
        else:
            rewritten, count = rewrite_device(job.cmd)
            if count:
                print(f'    -> auto gpu; {count} --device flag(s) retargeted to cuda:0')
                print(f'       {rewritten}')
            else:
                print('    -> auto gpu via CUDA_VISIBLE_DEVICES')
    if doc.staging:
        print(f'\n{len(doc.staging)} line(s) staged below the barrier, not checked')
    return 0


def cmd_check(args) -> int:
    """Show what the busy detector currently sees."""
    try:
        gpus = nvidia_smi_probe(args.floor_mib)
    except RuntimeError as exc:
        print(f'gpu probe failed: {exc}', file=sys.stderr)
        return 2
    state = load_state(Path(args.log_dir) / 'state.json')
    ours = {run.gpu: run for run in (state.running if state else [])}
    print(f'foreign memory floor: {args.floor_mib} MiB')
    for gpu in gpus:
        mine = ours.get(gpu.index)
        if mine is not None:
            print(f'  gpu {gpu.index}  OURS  #{mine.jid:04d} {mine.name} pid {mine.pid}')
        elif gpu.busy:
            print(
                f'  gpu {gpu.index}  BUSY  foreign pid(s) {gpu.foreign_pids} '
                f'({gpu.foreign_mib} MiB)'
            )
        else:
            print(f'  gpu {gpu.index}  free')
    return 0


def cmd_status(args) -> int:
    """Read the state file written by the watcher."""
    log_dir = Path(args.log_dir)
    state = load_state(log_dir / 'state.json')
    if state is None:
        print(f'no gpuq state in {log_dir} yet')
        return 0

    print(f'queue file : {state.queue_file}')
    if state.paused:
        print("PAUSED     : a '# pause' line is holding new launches")
    now = time.time()
    if state.running:
        print(f'running ({len(state.running)}):')
        for run in state.running:
            live = 'alive' if proc_alive(run.pid, run.starttime) else 'GONE'
            where = 'ALL' if run.gpu is None else f'gpu{run.gpu}'
            print(
                f'  #{run.jid:04d} {run.name}  {where}  pid {run.pid} [{live}]  '
                f'up {_age(now - run.started_at)}  {run.log_path}'
            )
    else:
        print('running : none')

    if state.pending:
        print(f'pending ({len(state.pending)}):')
        for job in state.pending[:20]:
            flags = ' [excl]' if job.exclusive else ''
            flags += f' [gpu={job.pin_gpu}]' if job.pin_gpu is not None else ''
            flags += f' [after={",".join(job.after)}]' if job.after else ''
            print(f'  #{job.jid:04d} {job.name}{flags}  {job.cmd[:90]}')
        if len(state.pending) > 20:
            print(f'  ... and {len(state.pending) - 20} more')
    else:
        print('pending : none')
    return 0


def cmd_kill(args) -> int:
    """Stop the job holding a GPU, and print how to put it back in the queue."""
    log_dir = Path(args.log_dir)
    state = load_state(log_dir / 'state.json')
    running = state.running if state else []
    try:
        gpus = nvidia_smi_probe(args.floor_mib)
    except RuntimeError as exc:
        print(f'note: gpu probe failed ({exc}), going on state.json alone', file=sys.stderr)
        gpus = []

    job, err = resolve_kill_target(args.target, running, gpus)
    if job is None:
        print(err, file=sys.stderr)
        return 2

    where = 'ALL gpus' if job.gpu is None else f'gpu {job.gpu}'
    print(
        f'about to kill #{job.jid:04d} {job.name} '
        f'({where}, pid {job.pid}, up {_age(time.time() - job.started_at)})'
    )
    print(f'  {job.cmd_as_run}')
    if not args.yes:
        try:
            answer = input('kill? [y/N] ').strip().lower()
        except EOFError:
            answer = ''
        if answer not in ('y', 'yes'):
            print('cancelled')
            return 1

    outcome = kill_group(job.pid, args.sigint_grace, args.sigterm_grace)
    print(f'[gpuq] stopped with {outcome}' if outcome.startswith('SIG') else f'[gpuq] {outcome}')

    # deliberately NOT touching state.json. A live watcher owns that file, and
    # writing it back from a read taken seconds ago can rewind the queue cursor
    # and replay work. The watcher notices the dead pid on its next tick.
    with open(log_dir / 'ledger.jsonl', 'a', encoding='utf-8') as handle:
        handle.write(
            json.dumps(
                {
                    'ts': round(time.time(), 3),
                    'event': 'killed',
                    'jid': job.jid,
                    'name': job.name,
                    'gpu': job.gpu,
                    'pid': job.pid,
                    'outcome': outcome,
                }
            )
            + '\n'
        )

    plain = requeue_line(job.name, job.cmd_as_run)
    with_resume = requeue_line(job.name, job.cmd_as_run, add_resume=True)
    print('\nrequeue with:')
    print(f'  {plain}')
    if with_resume != plain:
        print('or, if it had already checkpointed:')
        print(f'  {with_resume}')

    if args.requeue:
        line = with_resume if args.resume else plain
        queue_path = Path(state.queue_file) if state and state.queue_file else None
        if queue_path is None or not queue_path.exists():
            print('cannot append: no queue file recorded in state.json', file=sys.stderr)
            return 2
        with open(queue_path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')
        print(f'appended to {queue_path}')
    return 0


def cmd_run(args) -> int:
    """Run the watcher in the foreground."""
    queue_path = Path(args.queue_file)
    if ensure_queue_file(queue_path):
        print(f'[gpuq] created {queue_path} with a barrier; paste commands above it')
    watcher = Watcher(
        queue_file=queue_path,
        log_dir=args.log_dir,
        poll_secs=args.poll,
        settle_secs=args.settle,
        floor_mib=args.floor_mib,
        cwd=args.cwd,
        sigint_grace=args.sigint_grace,
        sigterm_grace=args.sigterm_grace,
    )
    try:
        watcher.run()
    except WatcherLockError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog='gpuq',
        description='queue shell commands and run them as GPUs free up',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'queue file lines:\n'
            '  python train.py --device cuda:0        auto gpu, --device retargeted\n'
            '  [excl] python train.py                 whole box, run verbatim\n'
            '  [gpu=1] python train.py                pinned to gpu 1, run verbatim\n'
            '  [name=foo] python train.py             explicit log name\n'
            '  [after=foo] python analyse.py          waits for foo to exit 0\n'
            '  # pause                                stop starting new jobs\n'
            '  # comment                              ignored\n'
            '  ####...                                barrier: nothing below it runs\n'
            '\n'
            'the watcher rewrites the file live: started jobs become\n'
            "'### RUNNING ON CUDA:N ###' lines, finished ones move to <queue>.done.txt.\n"
            'edit above the barrier to change the queue, draft below it.\n'
        ),
    )
    subs = parser.add_subparsers(dest='command', required=True)

    def add_common(sub):
        sub.add_argument(
            '--log-dir',
            default=DEFAULT_LOG_DIR,
            help=f'where logs and state live (default {DEFAULT_LOG_DIR})',
        )
        sub.add_argument(
            '--floor-mib',
            type=int,
            default=DEFAULT_FLOOR_MIB,
            help=f'ignore foreign cuda contexts smaller than this (default {DEFAULT_FLOOR_MIB})',
        )

    def add_grace(sub):
        sub.add_argument(
            '--sigint-grace',
            type=float,
            default=DEFAULT_SIGINT_GRACE,
            help='seconds after SIGINT before escalating to SIGTERM',
        )
        sub.add_argument(
            '--sigterm-grace',
            type=float,
            default=DEFAULT_SIGTERM_GRACE,
            help='seconds after SIGTERM before SIGKILL',
        )

    p_validate = subs.add_parser('validate', help='parse the queue file, launch nothing')
    p_validate.add_argument('queue_file')

    p_run = subs.add_parser('run', help='watch the queue and launch jobs')
    p_run.add_argument('queue_file')
    add_common(p_run)
    add_grace(p_run)
    p_run.add_argument(
        '--poll',
        type=float,
        default=DEFAULT_POLL_SECS,
        help=f'seconds between checks (default {DEFAULT_POLL_SECS:.0f})',
    )
    p_run.add_argument(
        '--settle',
        type=float,
        default=DEFAULT_SETTLE_SECS,
        help='seconds to let vram come back before an exclusive job',
    )
    p_run.add_argument('--cwd', default=None, help='working dir for jobs (default: here)')

    p_check = subs.add_parser('check', help='show what the busy detector sees')
    add_common(p_check)

    p_status = subs.add_parser('status', help='show queued and running jobs')
    add_common(p_status)

    p_kill = subs.add_parser('kill', help='stop the job holding a gpu, e.g. cuda:0')
    p_kill.add_argument('target', help='cuda:N, a gpu index, a job name, or #jid')
    add_common(p_kill)
    add_grace(p_kill)
    p_kill.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    p_kill.add_argument(
        '--requeue', action='store_true', help='append the job back onto the queue file'
    )
    p_kill.add_argument(
        '--resume', action='store_true', help='with --requeue, add --resume to the appended line'
    )

    args = parser.parse_args(argv)
    handlers = {
        'validate': cmd_validate,
        'run': cmd_run,
        'check': cmd_check,
        'status': cmd_status,
        'kill': cmd_kill,
    }
    return handlers[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
