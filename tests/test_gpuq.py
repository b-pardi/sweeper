"""Tests for gpuq.py, the GPU job queuer.

Run from the repo root:
    python -m pytest tests/test_gpuq.py -v

Nothing here touches a real GPU. The nvidia-smi probe is fed canned CSV text and
the scheduler is a pure function, so the whole scheduling core is testable
without a driver.
"""

import json
import os
import signal
import subprocess
import time

import pytest

import gpuq

# ---------------------------------------------------------------- queue parsing


def test_excl_directive_marks_job_exclusive():
    job = gpuq.parse_queue_line('[excl] python train.py', lineno=1, jid=1)
    assert job.exclusive is True
    assert job.cmd == 'python train.py'


def test_gpu_directive_pins_the_job():
    job = gpuq.parse_queue_line('[gpu=1] python train.py', lineno=1, jid=1)
    assert job.pin_gpu == 1
    assert job.cmd == 'python train.py'


def test_name_directive_sets_the_job_name():
    job = gpuq.parse_queue_line('[name=my-run] python train.py', lineno=1, jid=1)
    assert job.name == 'my-run'


def test_directives_combine_with_commas():
    job = gpuq.parse_queue_line('[excl,name=big] python train.py', lineno=1, jid=1)
    assert job.exclusive is True
    assert job.name == 'big'


def test_directive_whitespace_is_tolerated():
    job = gpuq.parse_queue_line('[ excl , name = big ] python train.py', lineno=1, jid=1)
    assert job.exclusive is True
    assert job.name == 'big'


def test_name_falls_back_to_exp_tag_from_the_command():
    job = gpuq.parse_queue_line('python sweep.py --exp-tag exp-final', lineno=1, jid=7)
    assert job.name == 'exp-final'


def test_name_falls_back_to_job_id_without_exp_tag():
    job = gpuq.parse_queue_line('python train.py', lineno=1, jid=7)
    assert job.name == 'job0007'


def test_explicit_name_beats_exp_tag():
    job = gpuq.parse_queue_line('[name=mine] python sweep.py --exp-tag other', lineno=1, jid=1)
    assert job.name == 'mine'


def test_unknown_directive_is_a_hard_error():
    with pytest.raises(gpuq.QueueParseError, match='exlc'):
        gpuq.parse_queue_line('[exlc] python train.py', lineno=3, jid=1)


def test_unclosed_bracket_is_a_hard_error():
    with pytest.raises(gpuq.QueueParseError, match=r'\]'):
        gpuq.parse_queue_line('[excl python train.py', lineno=3, jid=1)


def test_non_integer_gpu_is_a_hard_error():
    with pytest.raises(gpuq.QueueParseError, match='gpu'):
        gpuq.parse_queue_line('[gpu=cuda0] python train.py', lineno=3, jid=1)


def test_parse_error_reports_the_line_number():
    with pytest.raises(gpuq.QueueParseError, match='line 3'):
        gpuq.parse_queue_line('[exlc] python train.py', lineno=3, jid=1)


def test_comments_and_blanks_are_skipped():
    assert gpuq.parse_queue_line('# a comment', lineno=1, jid=1) is None
    assert gpuq.parse_queue_line('   ', lineno=1, jid=1) is None
    assert gpuq.parse_queue_line('', lineno=1, jid=1) is None


def test_parse_queue_file_numbers_jobs_skipping_comments():
    text = '\n'.join(
        [
            '# header',
            '',
            'python a.py',
            '# another comment',
            '[excl] python b.py',
        ]
    )
    jobs = gpuq.parse_queue_file(text, start_jid=1)
    assert [j.jid for j in jobs] == [1, 2]
    assert [j.lineno for j in jobs] == [3, 5]
    assert jobs[1].exclusive is True


@pytest.mark.parametrize(
    ('text', 'kwargs', 'expected'),
    [
        pytest.param(
            'python a.py\npython b.py', {'start_jid': 5}, [(5, 1), (6, 2)], id='start_jid'
        ),
        pytest.param(
            'python c.py', {'start_jid': 3, 'start_lineno': 5}, [(3, 5)], id='start_lineno'
        ),
    ],
)
def test_parse_queue_file_honours_start_jid(text, kwargs, expected):
    jobs = gpuq.parse_queue_file(text, **kwargs)
    assert [(j.jid, j.lineno) for j in jobs] == expected


# ------------------------------------------------------------- device rewriting


def test_rewrite_device_maps_cuda1_to_cuda0():
    cmd, n = gpuq.rewrite_device('python t.py --device cuda:1')
    assert cmd == 'python t.py --device cuda:0'
    assert n == 1


def test_rewrite_device_handles_the_equals_form():
    cmd, n = gpuq.rewrite_device('python t.py --device=cuda:1')
    assert cmd == 'python t.py --device=cuda:0'
    assert n == 1


@pytest.mark.parametrize(
    'command',
    [
        pytest.param('python t.py --device cuda:0', id='cuda0'),
        pytest.param('python t.py --epochs 5', id='no_device'),
        pytest.param('python t.py --device cpu', id='cpu'),
        pytest.param('python t.py --device-map cuda:1', id='longer_flag'),
    ],
)
def test_rewrite_device_leaves_commands_alone(command):
    cmd, n = gpuq.rewrite_device(command)
    assert cmd == command
    assert n == 0


def test_rewrite_device_ignores_cuda_outside_the_device_flag():
    # a path that happens to contain cuda:1 must survive untouched
    cmd, n = gpuq.rewrite_device('python t.py --out logs/cuda:1/run --device cuda:1')
    assert cmd == 'python t.py --out logs/cuda:1/run --device cuda:0'
    assert n == 1


def test_rewrite_device_rewrites_every_occurrence():
    cmd, n = gpuq.rewrite_device('python a.py --device cuda:1 && python b.py --device cuda:1')
    assert cmd.count('cuda:0') == 2
    assert n == 2


# ------------------------------------------------------------------- gpu probe

UUID_CSV = """index, uuid
0, GPU-aaa
1, GPU-bbb
"""

# the real shape of things on this box: one job on gpu0, one on gpu1 that also
# left a small stray context on gpu0 during device enumeration
APPS_CSV = """pid, gpu_uuid, used_gpu_memory [MiB]
2119414, GPU-aaa, 906 MiB
2119620, GPU-aaa, 338 MiB
2119620, GPU-bbb, 1078 MiB
"""


def test_probe_reports_one_state_per_gpu():
    gpus = gpuq.probe_gpus(UUID_CSV, APPS_CSV, floor_mib=400)
    assert [g.index for g in gpus] == [0, 1]


def test_probe_ignores_contexts_below_the_memory_floor():
    gpus = gpuq.probe_gpus(UUID_CSV, APPS_CSV, floor_mib=400)
    assert gpus[0].foreign_pids == [2119414]  # the 338 MiB stray context is dropped
    assert gpus[0].foreign_mib == 906


def test_probe_counts_a_pid_that_spans_two_gpus():
    gpus = gpuq.probe_gpus(UUID_CSV, APPS_CSV, floor_mib=400)
    assert gpus[1].foreign_pids == [2119620]
    assert gpus[1].foreign_mib == 1078


def test_probe_marks_gpu_busy_above_floor():
    gpus = gpuq.probe_gpus(UUID_CSV, APPS_CSV, floor_mib=400)
    assert all(g.busy for g in gpus)


def test_probe_with_no_compute_apps_leaves_every_gpu_free():
    gpus = gpuq.probe_gpus(UUID_CSV, 'pid, gpu_uuid, used_gpu_memory [MiB]\n', floor_mib=400)
    assert [g.busy for g in gpus] == [False, False]


def test_probe_treats_unreadable_memory_as_busy():
    # nvidia-smi emits [N/A] when it cannot read a process; we cannot prove the
    # gpu is idle, so fail safe and call it busy
    apps = 'pid, gpu_uuid, used_gpu_memory [MiB]\n999, GPU-aaa, [N/A]\n'
    gpus = gpuq.probe_gpus(UUID_CSV, apps, floor_mib=400)
    assert gpus[0].busy is True
    assert gpus[1].busy is False


def test_a_lone_small_job_is_busy_not_noise():
    # a small training job sat at 388 MiB and a 400 MiB flat floor called gpu 1 idle,
    # so gpuq would schedule on top of it
    apps = 'pid, gpu_uuid, used_gpu_memory [MiB]\n2340386, GPU-bbb, 388 MiB\n'
    gpus = gpuq.probe_gpus(UUID_CSV, apps, floor_mib=gpuq.DEFAULT_FLOOR_MIB)
    assert gpus[1].busy is True
    assert gpus[1].foreign_pids == [2340386]


def test_a_stray_context_is_reported_as_ignored():
    gpus = gpuq.probe_gpus(UUID_CSV, APPS_CSV, floor_mib=1)
    assert (2119620, 338) in gpus[0].ignored


def test_a_job_spread_evenly_across_gpus_occupies_both():
    # real multi-gpu training allocates comparably everywhere, unlike a stray context
    apps = 'pid, gpu_uuid, used_gpu_memory [MiB]\n555, GPU-aaa, 5000 MiB\n555, GPU-bbb, 4800 MiB\n'
    gpus = gpuq.probe_gpus(UUID_CSV, apps, floor_mib=gpuq.DEFAULT_FLOOR_MIB)
    assert [g.busy for g in gpus] == [True, True]


def test_probe_marks_everything_busy_when_a_uuid_is_unmappable():
    # an app on a gpu we cannot identify means our view is incomplete, so no gpu
    # may be considered free
    apps = 'pid, gpu_uuid, used_gpu_memory [MiB]\n999, GPU-zzz, 8000 MiB\n'
    gpus = gpuq.probe_gpus(UUID_CSV, apps, floor_mib=400)
    assert [g.busy for g in gpus] == [True, True]


# --------------------------------------------------------------- proc liveness


def test_parse_proc_stat_extracts_pgrp_and_starttime():
    fields = ['1234', '(python3)', 'S', '1000', '4321'] + ['0'] * 16 + ['987654']
    assert gpuq.parse_proc_stat(' '.join(fields)) == (4321, 987654)


def test_parse_proc_stat_survives_parens_and_spaces_in_comm():
    # a process named "we ird) name" would shift every field on a naive split
    fields = ['1234', '(we ird) name)', 'S', '1000', '4321'] + ['0'] * 16 + ['987654']
    assert gpuq.parse_proc_stat(' '.join(fields)) == (4321, 987654)


def test_proc_starttime_returns_none_for_a_dead_pid():
    assert gpuq.proc_starttime(4_000_000) is None


def test_proc_alive_accepts_the_matching_starttime():
    pid = os.getpid()
    assert gpuq.proc_alive(pid, gpuq.proc_starttime(pid)) is True


def test_proc_alive_rejects_a_recycled_pid():
    # same pid, different start time: this is a different process wearing the
    # same number, and adopting it would be a serious bug
    pid = os.getpid()
    assert gpuq.proc_alive(pid, gpuq.proc_starttime(pid) + 1) is False


def test_proc_alive_is_false_for_a_dead_pid():
    assert gpuq.proc_alive(4_000_000, 12345) is False


def test_group_members_finds_our_own_process_group():
    members = gpuq.group_members(os.getpgid(0))
    assert os.getpid() in members


# ------------------------------------------------------------- scheduler core


def mkjob(jid=1, exclusive=False, pin_gpu=None, cmd='python t.py'):
    return gpuq.Job(
        jid=jid,
        name=f'job{jid}',
        cmd=cmd,
        exclusive=exclusive,
        pin_gpu=pin_gpu,
        raw_line=cmd,
        lineno=jid,
    )


def mkrunning(jid=1, gpu=0, exclusive=False):
    return gpuq.RunningJob(
        jid=jid,
        name=f'job{jid}',
        gpu=gpu,
        pid=1000 + jid,
        starttime=1,
        exclusive=exclusive,
        log_path='/dev/null',
        cmd_as_run='python t.py',
        started_at=0.0,
    )


def mkgpus(busy=(False, False)):
    return [gpuq.GpuState(index=i, busy=b) for i, b in enumerate(busy)]


def decide(
    pending, running=(), gpus=None, now=1000.0, last_exit_at=0.0, settle_secs=10.0, finished=None
):
    return gpuq.decide(
        pending=list(pending),
        running=list(running),
        gpus=gpus if gpus is not None else mkgpus(),
        now=now,
        last_exit_at=last_exit_at,
        settle_secs=settle_secs,
        finished=dict(finished or {}),
    )


def test_empty_queue_waits():
    assert decide([]).action == 'wait'


def test_first_job_takes_the_lowest_free_gpu():
    d = decide([mkjob()])
    assert d.action == 'launch'
    assert d.gpu == 0


def test_job_skips_a_gpu_held_by_a_foreign_process():
    d = decide([mkjob()], gpus=mkgpus(busy=(True, False)))
    assert (d.action, d.gpu) == ('launch', 1)


def test_job_waits_when_every_gpu_is_foreign_busy():
    assert decide([mkjob()], gpus=mkgpus(busy=(True, True))).action == 'wait'


def test_job_skips_a_gpu_claimed_by_one_of_our_own_running_jobs():
    # our own bookkeeping is authoritative: a job launched seconds ago holds
    # gpu 0 even though it has not allocated any vram yet
    d = decide([mkjob(jid=2)], running=[mkrunning(jid=1, gpu=0)])
    assert (d.action, d.gpu) == ('launch', 1)


def test_job_waits_when_we_already_hold_every_gpu():
    running = [mkrunning(jid=1, gpu=0), mkrunning(jid=2, gpu=1)]
    assert decide([mkjob(jid=3)], running=running).action == 'wait'


def test_exclusive_job_waits_for_running_jobs_to_drain():
    d = decide([mkjob(exclusive=True)], running=[mkrunning(gpu=0)])
    assert d.action == 'wait'
    assert 'drain' in d.reason.lower()


def test_exclusive_job_launches_alone_once_the_box_is_clear():
    d = decide([mkjob(exclusive=True)], now=1000.0, last_exit_at=900.0)
    assert d.action == 'launch'
    assert d.gpu is None  # all gpus visible, command untouched


def test_exclusive_job_waits_on_foreign_gpu_activity():
    d = decide([mkjob(exclusive=True)], gpus=mkgpus(busy=(False, True)))
    assert d.action == 'wait'


def test_exclusive_job_waits_out_the_settle_window():
    # vram takes a moment to come back after the last job exits
    d = decide([mkjob(exclusive=True)], now=1000.0, last_exit_at=995.0, settle_secs=10.0)
    assert d.action == 'wait'
    assert 'settl' in d.reason.lower()


def test_nothing_starts_while_an_exclusive_job_runs():
    d = decide([mkjob(jid=2)], running=[mkrunning(jid=1, gpu=None, exclusive=True)])
    assert d.action == 'wait'


def test_queue_is_strict_fifo_and_does_not_jump_an_exclusive_head():
    pending = [mkjob(jid=1, exclusive=True), mkjob(jid=2)]
    d = decide(pending, running=[mkrunning(jid=9, gpu=0)])
    assert d.action == 'wait'  # job 2 must not overtake the exclusive head


def test_pinned_job_takes_its_own_gpu():
    d = decide([mkjob(pin_gpu=1)])
    assert (d.action, d.gpu) == ('launch', 1)


def test_pinned_job_waits_for_its_gpu_rather_than_using_a_free_one():
    d = decide([mkjob(pin_gpu=0)], gpus=mkgpus(busy=(True, False)))
    assert d.action == 'wait'


def test_wait_reasons_name_the_job_rather_than_a_churning_id():
    # queued jobs have no durable id until launch, so a reason built from jid
    # would rename the same job every tick
    d = decide([mkjob(jid=99, exclusive=True)], running=[mkrunning(gpu=0)])
    assert 'job99' in d.reason


def test_pin_to_a_nonexistent_gpu_is_an_error_not_a_deadlock():
    d = decide([mkjob(pin_gpu=7)])
    assert d.action == 'error'
    assert '7' in d.reason


# ------------------------------------------------- state + append-only cursor


def test_state_round_trips_through_disk(tmp_path):
    path = tmp_path / 'state.json'
    state = gpuq.State(
        queue_file='q.txt',
        next_jid=9,
        pending=[mkjob(jid=9, exclusive=True)],
        running=[mkrunning(jid=8, gpu=1)],
        last_exit_at=123.5,
    )
    gpuq.save_state(path, state)
    back = gpuq.load_state(path)

    assert back.next_jid == 9
    assert back.last_exit_at == 123.5
    assert back.pending[0].exclusive is True
    assert back.running[0].gpu == 1
    assert isinstance(back.pending[0], gpuq.Job)
    assert isinstance(back.running[0], gpuq.RunningJob)


def test_load_state_returns_none_when_there_is_nothing_yet(tmp_path):
    assert gpuq.load_state(tmp_path / 'state.json') is None


def test_save_state_leaves_no_temp_file_behind(tmp_path):
    gpuq.save_state(tmp_path / 'state.json', gpuq.State())
    assert [p.name for p in tmp_path.iterdir()] == ['state.json']


def test_save_state_replaces_atomically(tmp_path):
    # a reader must never see a half written file, so the old content has to
    # stay intact until the rename lands
    path = tmp_path / 'state.json'
    gpuq.save_state(path, gpuq.State(next_jid=1))
    gpuq.save_state(path, gpuq.State(next_jid=2))
    assert gpuq.load_state(path).next_jid == 2


# ------------------------------------------------------- reattach + kill


@pytest.fixture
def spawn():
    """Start throwaway processes in their own session, guaranteed cleaned up."""
    started = []

    def _spawn(argv):
        proc = subprocess.Popen(argv, start_new_session=True)
        started.append(proc)
        return proc

    yield _spawn

    for proc in started:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def running_for(proc, starttime=None):
    return gpuq.RunningJob(
        jid=1,
        name='probe',
        gpu=0,
        pid=proc.pid,
        starttime=gpuq.proc_starttime(proc.pid) if starttime is None else starttime,
        exclusive=False,
        log_path='/dev/null',
        cmd_as_run='sleep',
        started_at=0.0,
    )


def test_group_members_ignores_a_zombie(spawn):
    # a killed but unreaped child still has a /proc entry and keeps its pgid;
    # counting it as alive would make kill_group spin until timeout
    proc = spawn(['sleep', '30'])
    os.killpg(proc.pid, signal.SIGKILL)
    _wait_until(
        lambda: gpuq.proc_starttime(proc.pid) is None or _proc_is_zombie(proc.pid), timeout=5
    )
    assert gpuq.group_members(proc.pid) == []


def _proc_is_zombie(pid):
    try:
        with open(f'/proc/{pid}/stat', encoding='utf-8') as handle:
            return handle.read().rpartition(')')[2].split()[0] == 'Z'
    except OSError:
        return False


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_reattach_adopts_a_job_that_is_still_alive(spawn):
    proc = spawn(['sleep', '30'])
    result = gpuq.reattach([running_for(proc)])
    assert [j.pid for j in result.alive] == [proc.pid]
    assert result.alive[0].adopted is True
    assert result.dead == []


def test_reattach_drops_a_job_that_ended_while_we_were_away(spawn):
    proc = spawn(['sleep', '30'])
    job = running_for(proc)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)
    result = gpuq.reattach([job])
    assert result.alive == []
    assert [j.pid for j in result.dead] == [proc.pid]


def test_reattach_rejects_a_recycled_pid(spawn):
    # right pid, wrong start time: a different process wearing the same number
    proc = spawn(['sleep', '30'])
    job = running_for(proc, starttime=gpuq.proc_starttime(proc.pid) + 1)
    result = gpuq.reattach([job])
    assert result.alive == []
    assert [j.pid for j in result.dead] == [proc.pid]


def test_reattach_warns_about_leftovers_but_never_adopts_them(spawn):
    # leader exec'd away leaving a worker behind: we cannot prove the group is
    # ours any more, so report it and let the foreign gpu check handle it
    proc = spawn(['sh', '-c', 'sleep 30 & wait'])
    job = running_for(proc)
    _wait_until(lambda: len(gpuq.group_members(proc.pid)) >= 2)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)
    result = gpuq.reattach([job])
    assert result.alive == []
    assert result.orphans and result.orphans[0][0].pid == proc.pid


def test_kill_group_reports_when_the_group_is_already_gone(spawn):
    proc = spawn(['sleep', '30'])
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)
    assert gpuq.kill_group(proc.pid, sigint_grace=0.4, sigterm_grace=0.4) == 'already gone'


def test_kill_group_stops_at_sigint_when_that_is_enough(spawn):
    proc = spawn(['sleep', '30'])
    assert gpuq.kill_group(proc.pid, sigint_grace=3.0, sigterm_grace=3.0) == 'SIGINT'


def test_kill_group_escalates_to_sigterm(spawn, tmp_path):
    script = tmp_path / 'ignore_int.py'
    script.write_text(
        'import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(60)\n'
    )
    proc = spawn(['python3', str(script)])
    _wait_until(lambda: gpuq.group_members(proc.pid) != [])
    assert gpuq.kill_group(proc.pid, sigint_grace=0.5, sigterm_grace=3.0) == 'SIGTERM'


def test_kill_group_escalates_to_sigkill_for_a_wedged_job(spawn, tmp_path):
    script = tmp_path / 'ignore_both.py'
    script.write_text(
        'import signal, time\n'
        'signal.signal(signal.SIGINT, signal.SIG_IGN)\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        'time.sleep(60)\n'
    )
    proc = spawn(['python3', str(script)])
    _wait_until(lambda: gpuq.group_members(proc.pid) != [])
    assert gpuq.kill_group(proc.pid, sigint_grace=0.5, sigterm_grace=0.5) == 'SIGKILL'


def test_kill_group_takes_down_child_processes_too(spawn):
    proc = spawn(['sh', '-c', 'sleep 60 & sleep 60'])
    _wait_until(lambda: len(gpuq.group_members(proc.pid)) >= 2)
    gpuq.kill_group(proc.pid, sigint_grace=0.5, sigterm_grace=0.5)
    assert gpuq.group_members(proc.pid) == []


# ------------------------------------------------------------- kill targeting


@pytest.mark.parametrize(
    'target', [pytest.param('cuda:1', id='cuda_string'), pytest.param('1', id='bare_index')]
)
def test_kill_target_resolves_a_gpu_index(target):
    running = [mkrunning(jid=4, gpu=1)]
    job, err = gpuq.resolve_kill_target(target, running, mkgpus())
    assert err is None and job.jid == 4


def test_kill_target_resolves_a_job_name():
    running = [mkrunning(jid=4, gpu=1)]
    job, err = gpuq.resolve_kill_target('job4', running, mkgpus())
    assert err is None and job.jid == 4


def test_kill_target_matches_an_exclusive_job_on_any_gpu():
    running = [mkrunning(jid=4, gpu=None, exclusive=True)]
    job, err = gpuq.resolve_kill_target('cuda:0', running, mkgpus())
    assert err is None and job.jid == 4


def test_kill_target_refuses_a_process_we_did_not_launch():
    gpus = [
        gpuq.GpuState(index=0, foreign_pids=[2119414], foreign_mib=906, busy=True),
        gpuq.GpuState(index=1),
    ]
    job, err = gpuq.resolve_kill_target('cuda:0', [], gpus)
    assert job is None
    assert '2119414' in err and 'did not launch' in err


def test_kill_target_reports_an_idle_gpu():
    job, err = gpuq.resolve_kill_target('cuda:0', [], mkgpus())
    assert job is None and 'no gpuq job' in err.lower()


def test_kill_target_rejects_gibberish():
    job, err = gpuq.resolve_kill_target('banana', [], mkgpus())
    assert job is None and err is not None


# ----------------------------------------------------- watcher integration
#
# Real subprocesses, fake nvidia-smi. tick() is driven by hand so the tests are
# deterministic instead of racing a poll interval.


class FakeProbe:
    """Stand-in for nvidia-smi. Flip .busy to simulate foreign jobs."""

    def __init__(self, n_gpus=2):
        self.busy = [False] * n_gpus
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return [
            gpuq.GpuState(index=i, busy=b, foreign_pids=[9999] if b else [])
            for i, b in enumerate(self.busy)
        ]


@pytest.fixture
def watcher_factory(tmp_path):
    made = []

    def _make(queue_text='', n_gpus=2, settle_secs=0.0, **kwargs):
        queue = tmp_path / f'queue{len(made)}.txt'
        queue.write_text(queue_text)
        probe = FakeProbe(n_gpus)
        watcher = gpuq.Watcher(
            queue_file=queue,
            log_dir=tmp_path / f'logs{len(made)}',
            gpu_probe=probe,
            settle_secs=settle_secs,
            cwd=str(tmp_path),
            **kwargs,
        )
        watcher.probe = probe
        watcher.queue_path = queue
        made.append(watcher)
        return watcher

    yield _make

    for watcher in made:
        watcher.shutdown_hard()


def ledger_events(watcher):
    path = watcher.log_dir / 'ledger.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tick_until(watcher, predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        watcher.tick()
        if predicate():
            return True
        time.sleep(0.05)
    return False


def drain(watcher, timeout=20.0):
    """Tick until nothing is pending or running."""
    return tick_until(
        watcher, lambda: not watcher.state.pending and not watcher.state.running, timeout
    )


def test_two_jobs_run_concurrently_on_two_gpus(watcher_factory):
    w = watcher_factory('sleep 2\nsleep 2\n')
    w.tick()
    assert sorted(r.gpu for r in w.state.running) == [0, 1]


def test_third_job_waits_until_a_gpu_frees_up(watcher_factory):
    w = watcher_factory('sleep 2\nsleep 2\nsleep 0.1\n')
    w.tick()
    assert len(w.state.running) == 2
    assert len(w.state.pending) == 1
    assert drain(w), 'queue never drained'
    assert [e['jid'] for e in ledger_events(w) if e['event'] == 'started'] == [1, 2, 3]


def test_exclusive_job_gets_every_gpu_and_an_unmodified_command(watcher_factory):
    w = watcher_factory('[excl] echo "cvd=[$CUDA_VISIBLE_DEVICES] dev=--device cuda:1"\n')
    assert drain(w)
    log = next(w.log_dir.glob('0001-*.log')).read_text()
    assert 'cvd=[]' in log  # all gpus visible
    assert 'dev=--device cuda:1' in log  # command left alone


# ----------------------------------------------------------- launch mechanics


def test_auto_assigned_job_gets_cuda_visible_devices_and_a_rewritten_device(watcher_factory):
    w = watcher_factory('echo "cvd=$CUDA_VISIBLE_DEVICES"\n')
    w.probe.busy = [True, False]  # force it onto gpu 1
    assert drain(w)
    log = next(w.log_dir.glob('0001-*.log')).read_text()
    assert 'cvd=1' in log


def test_pinned_job_runs_verbatim_on_its_own_gpu(watcher_factory):
    w = watcher_factory('[gpu=1] echo "cvd=[$CUDA_VISIBLE_DEVICES] dev=--device cuda:1"\n')
    assert drain(w)
    log = next(w.log_dir.glob('0001-*.log')).read_text()
    assert 'cvd=[]' in log
    assert 'dev=--device cuda:1' in log


def test_job_name_drives_the_log_file_name(watcher_factory):
    w = watcher_factory('python3 -c pass --exp-tag exp-final\n')
    w.tick()
    assert (w.log_dir / '0001-exp-final.log').exists()


# --------------------------------------------------------------- logging


def test_log_has_a_header_and_a_footer_with_the_exit_code(watcher_factory):
    w = watcher_factory('echo hello from the job\n')
    assert drain(w)
    log = next(w.log_dir.glob('0001-*.log')).read_text()
    assert 'hello from the job' in log
    assert 'echo hello from the job' in log  # header echoes the command
    assert 'exit_code=0' in log


def test_ledger_start_record_carries_the_command_actually_run(watcher_factory):
    w = watcher_factory('echo x --device cuda:1\n')
    w.probe.busy = [True, False]
    w.tick()
    started = [e for e in ledger_events(w) if e['event'] == 'started'][0]
    assert started['cmd_as_run'] == 'echo x --device cuda:0'
    assert started['gpu'] == 1
    assert started['cuda_visible_devices'] == '1'


# ---------------------------------------------------- queue document + barrier
#
# The watcher owns the queue file. Lines above the #### barrier are live; the
# region below it is a staging area you can paste into and fine tune, and the
# watcher never touches it.

BAR = '########'


def test_a_line_of_four_or_more_hashes_is_the_barrier():
    doc = gpuq.parse_queue_document(f'python a.py\n{BAR}\npython b.py\n')
    assert doc.content == ['python a.py']
    assert doc.staging == ['python b.py']


def test_ordinary_comments_are_not_barriers():
    doc = gpuq.parse_queue_document('# note\n## note\n### note\npython a.py\n')
    assert doc.staging == []
    assert 'python a.py' in doc.content


def test_a_running_marker_is_three_hashes_so_it_never_reads_as_a_barrier():
    doc = gpuq.parse_queue_document('### RUNNING ON CUDA:0 ### python a.py\npython b.py\n')
    assert doc.staging == []
    assert doc.content == ['python b.py']  # the marker line is not queued work


def test_the_first_barrier_wins():
    doc = gpuq.parse_queue_document(f'python a.py\n{BAR}\npython b.py\n{BAR}\npython c.py\n')
    assert doc.content == ['python a.py']
    assert doc.staging == ['python b.py', BAR, 'python c.py']


def test_a_file_with_no_barrier_is_entirely_live():
    doc = gpuq.parse_queue_document('python a.py\npython b.py\n')
    assert doc.content == ['python a.py', 'python b.py']
    assert doc.staging == []
    assert doc.barrier is None


def test_comments_above_the_barrier_are_kept_as_content():
    doc = gpuq.parse_queue_document(f'# sweep 16.3\npython a.py\n{BAR}\n')
    assert doc.content == ['# sweep 16.3', 'python a.py']


def test_rendering_puts_running_markers_first_then_content_then_staging():
    text = gpuq.render_queue_document(
        running=[mkrunning(jid=1, gpu=1)],
        content=['# note', 'python b.py'],
        staging=['python draft.py'],
    )
    lines = text.splitlines()
    assert lines[0].startswith('### RUNNING ON CUDA:1 ###')
    assert '# note' in lines
    assert lines.index('# note') > 0
    barrier_at = next(i for i, ln in enumerate(lines) if ln.startswith('####'))
    assert lines.index('python b.py') < barrier_at
    assert lines.index('python draft.py') > barrier_at


def test_an_exclusive_job_renders_as_running_on_all_gpus():
    text = gpuq.render_queue_document(
        running=[mkrunning(jid=1, gpu=None, exclusive=True)],
        content=[],
        staging=[],
    )
    assert '### RUNNING ON ALL GPUS ###' in text


def test_a_rendered_running_line_keeps_the_original_directives():
    run = mkrunning(jid=1, gpu=0)
    run.raw_line = '[excl,name=big] python a.py'
    text = gpuq.render_queue_document(running=[run], content=[], staging=[])
    assert '[excl,name=big] python a.py' in text


def test_done_entry_carries_timestamp_exit_duration_and_gpu():
    entry = gpuq.done_entry(mkrunning(jid=1, gpu=1), exit_code=3, duration=2520.0)
    assert 'exit=3' in entry
    assert 'cuda:1' in entry
    assert '42m' in entry
    assert 'python t.py' in entry


def test_done_entry_marks_an_unknown_exit_code():
    entry = gpuq.done_entry(mkrunning(jid=1, gpu=0), exit_code=None, duration=5.0)
    assert 'exit=unknown' in entry


# --------------------------------------------- barrier behaviour in the watcher


def test_a_job_below_the_barrier_never_runs(watcher_factory):
    w = watcher_factory(f'{BAR}\nsleep 0.1\n')
    w.tick()
    assert w.state.running == []
    assert w.state.pending == []


def test_moving_a_job_above_the_barrier_makes_it_run(watcher_factory):
    w = watcher_factory(f'{BAR}\nsleep 0.1\n')
    w.tick()
    assert w.state.running == []
    w.queue_path.write_text(f'sleep 0.1\n{BAR}\n')
    w.tick()
    assert len(w.state.running) == 1


def test_the_staging_area_is_preserved_word_for_word(watcher_factory):
    staging = 'sleep 5 --half-written\n# my notes\n\n  indented draft\n'
    w = watcher_factory(f'sleep 0.1\n{BAR}\n{staging}')
    assert drain(w)
    doc = gpuq.parse_queue_document(w.queue_path.read_text())
    assert doc.staging == staging.splitlines()


def test_a_started_job_is_annotated_in_the_queue_file(watcher_factory):
    w = watcher_factory('sleep 5\n')
    w.probe.busy = [True, False]  # force gpu 1
    w.tick()
    assert '### RUNNING ON CUDA:1 ### sleep 5' in w.queue_path.read_text()


def test_a_started_job_stops_being_queued_work(watcher_factory):
    w = watcher_factory('sleep 5\nsleep 5\nsleep 5\n')
    w.tick()  # two of three start
    doc = gpuq.parse_queue_document(w.queue_path.read_text())
    assert doc.content == ['sleep 5']  # only the one still waiting


def test_a_finished_job_moves_to_the_done_file(watcher_factory):
    w = watcher_factory('echo hi\n')
    assert drain(w)
    assert 'echo hi' not in w.queue_path.read_text()
    done = w.done_path.read_text()
    assert 'echo hi' in done
    assert 'exit=0' in done


def test_done_entries_accumulate_in_order(watcher_factory):
    w = watcher_factory('echo first\n')
    assert drain(w)
    w.queue_path.write_text(f'echo second\n{BAR}\n')
    assert drain(w)
    done = w.done_path.read_text()
    assert done.index('echo first') < done.index('echo second')


def test_the_watcher_adds_a_barrier_to_a_file_that_has_none(watcher_factory):
    w = watcher_factory('sleep 5\n')
    w.tick()
    assert gpuq.parse_queue_document(w.queue_path.read_text()).barrier is not None


def test_deleting_a_queued_line_unqueues_it(watcher_factory):
    w = watcher_factory('sleep 5\nsleep 5\nsleep 9\n')
    w.tick()
    assert len(w.state.pending) == 1
    doc = gpuq.parse_queue_document(w.queue_path.read_text())
    w.queue_path.write_text(gpuq.render_queue_document(w.state.running, [], doc.staging))
    w.tick()
    assert w.state.pending == []


def test_a_clobbered_running_marker_does_not_double_launch(watcher_factory):
    # saving a stale editor buffer can put a running job back as a plain line;
    # relaunching it would put two trainings on one gpu
    w = watcher_factory('sleep 5\n')
    w.tick()
    pid = w.state.running[0].pid
    w.queue_path.write_text('sleep 5\n')  # marker wiped, plain line restored
    w.tick()
    assert [r.pid for r in w.state.running] == [pid]
    assert w.state.pending == []
    assert w.last_warning is not None and 'running' in w.last_warning.lower()
    assert '### RUNNING ON CUDA:0 ### sleep 5' in w.queue_path.read_text()


def test_a_queued_line_identical_to_a_running_one_still_queues(watcher_factory):
    # a running job lives in the file as a marker, so an identical plain line
    # underneath is genuinely new work and must not be mistaken for a wiped marker
    w = watcher_factory('sleep 5\nsleep 5\n')
    w.probe.busy = [False, True]  # one gpu, so the second line has to wait
    w.tick()
    assert len(w.state.running) == 1
    w.tick()
    assert len(w.state.pending) == 1
    assert w.last_warning is None


def test_a_bad_directive_above_the_barrier_is_reported_not_run(watcher_factory):
    w = watcher_factory('[exlc] sleep 1\n')
    w.tick()
    assert w.state.running == []
    assert w.last_warning is not None and 'exlc' in w.last_warning


def test_the_watcher_does_not_rewrite_a_file_that_has_not_changed(watcher_factory):
    w = watcher_factory(f'sleep 5\n{BAR}\n')
    w.tick()
    first = w.queue_path.read_text()
    mtime = w.queue_path.stat().st_mtime_ns
    w.tick()
    assert w.queue_path.read_text() == first
    assert w.queue_path.stat().st_mtime_ns == mtime


# ----------------------------------------------- live queue file + restart


def test_lines_added_above_the_barrier_while_running_are_picked_up(watcher_factory):
    w = watcher_factory('sleep 0.2\n')
    w.tick()
    w.queue_path.write_text('sleep 0.2\nsleep 0.2\n')
    assert drain(w)
    assert [e['jid'] for e in ledger_events(w) if e['event'] == 'started'] == [1, 2]


def test_a_restarted_watcher_reattaches_instead_of_relaunching(watcher_factory):
    w = watcher_factory('sleep 3\n')
    w.tick()
    pid = w.state.running[0].pid
    w.detach()  # walk away, leaving the job alive

    w2 = gpuq.Watcher(
        queue_file=w.queue_path,
        log_dir=w.log_dir,
        gpu_probe=w.probe,
        settle_secs=0.0,
        cwd=w.cwd,
    )
    w2.tick()
    assert [r.pid for r in w2.state.running] == [pid]
    assert w2.state.running[0].adopted is True
    assert len([e for e in ledger_events(w2) if e['event'] == 'started']) == 1
    w2.shutdown_hard()


def test_a_restarted_watcher_does_not_rerun_finished_work(watcher_factory):
    w = watcher_factory('true\n')
    assert drain(w)
    w2 = gpuq.Watcher(
        queue_file=w.queue_path,
        log_dir=w.log_dir,
        gpu_probe=w.probe,
        settle_secs=0.0,
        cwd=w.cwd,
    )
    w2.tick()
    assert w2.state.pending == []
    assert w2.state.running == []
    assert len([e for e in ledger_events(w2) if e['event'] == 'started']) == 1


def test_a_second_watcher_refuses_to_start_on_the_same_log_dir(watcher_factory):
    w = watcher_factory('')
    w.acquire_lock()
    w2 = gpuq.Watcher(queue_file=w.queue_path, log_dir=w.log_dir, gpu_probe=w.probe)
    with pytest.raises(gpuq.WatcherLockError):
        w2.acquire_lock()


def test_a_stale_lock_from_a_dead_watcher_is_taken_over(watcher_factory):
    w = watcher_factory('')
    (w.log_dir).mkdir(parents=True, exist_ok=True)
    (w.log_dir / 'watcher.lock').write_text(json.dumps({'pid': 4_000_000, 'starttime': 1}))
    w.acquire_lock()  # must not raise


# ------------------------------------------------------------ error paths


def test_a_job_pinned_to_a_missing_gpu_fails_out_instead_of_blocking(watcher_factory):
    w = watcher_factory('[gpu=7] echo nope\nsleep 0.1\n')
    assert drain(w)
    events = ledger_events(w)
    assert [e['event'] for e in events] == ['error', 'started', 'finished']
    assert '7' in [e for e in events if e['event'] == 'error'][0]['reason']
    # and it leaves the queue file, so it does not error again every tick
    assert 'gpu=7' not in w.queue_path.read_text()
    assert 'gpu=7' in w.done_path.read_text()


def test_draining_stops_new_launches_but_lets_running_jobs_finish(watcher_factory):
    w = watcher_factory('sleep 0.3\nsleep 5\n')
    w.probe.busy = [False, True]  # only one gpu, so job 2 stays queued
    w.tick()
    assert len(w.state.running) == 1
    w.begin_drain()
    assert tick_until(w, lambda: not w.state.running, timeout=10)
    assert len(w.state.pending) == 1  # job 2 never started
    assert not any(e['event'] == 'started' and e['jid'] == 2 for e in ledger_events(w))


def test_a_broken_nvidia_smi_stalls_the_queue_instead_of_guessing(watcher_factory):
    # if we cannot see the gpus we must not launch anything
    w = watcher_factory('sleep 5\n')

    def boom():
        raise RuntimeError('nvidia-smi exploded')

    w.gpu_probe = boom
    w.tick()
    assert w.state.running == []
    assert len(w.state.pending) == 1
    assert w.last_warning is not None and 'nvidia-smi' in w.last_warning


def test_first_interrupt_drains_and_the_second_detaches(watcher_factory):
    w = watcher_factory('')
    w.on_interrupt()
    assert w.draining is True and w.detached is False
    w.on_interrupt()
    assert w.detached is True


def test_proc_alive_treats_a_zombie_as_gone(spawn):
    # an exited-but-unreaped child keeps its /proc entry and its start time, so
    # a naive check would babysit a job that is already over
    proc = spawn(['sleep', '30'])
    starttime = gpuq.proc_starttime(proc.pid)
    os.killpg(proc.pid, signal.SIGKILL)
    assert _wait_until(lambda: _proc_is_zombie(proc.pid))
    assert gpuq.proc_alive(proc.pid, starttime) is False


# ------------------------------------------------------------------- cli


@pytest.mark.parametrize(
    ('command', 'add_resume', 'expected'),
    [
        pytest.param('python a.py', False, '[name=calo] python a.py', id='wraps_with_name'),
        pytest.param('python a.py', True, '[name=calo] python a.py --resume', id='adds_resume'),
        pytest.param(
            'python a.py --resume', True, '[name=calo] python a.py --resume', id='no_double_resume'
        ),
    ],
)
def test_requeue_line_wraps_the_command_with_its_name(command, add_resume, expected):
    assert gpuq.requeue_line('calo', command, add_resume=add_resume) == expected


def test_validate_lists_every_job(tmp_path, capsys):
    queue = tmp_path / 'q.txt'
    queue.write_text('# note\npython a.py\n[excl] python b.py --exp-tag beta\n')
    assert gpuq.main(['validate', str(queue)]) == 0
    out = capsys.readouterr().out
    assert 'python a.py' in out
    assert 'beta' in out
    assert 'excl' in out.lower()


def test_validate_warns_that_a_device_flag_will_be_retargeted(tmp_path, capsys):
    queue = tmp_path / 'q.txt'
    queue.write_text('python a.py --device cuda:1\n')
    assert gpuq.main(['validate', str(queue)]) == 0
    assert 'cuda:0' in capsys.readouterr().out


def test_validate_does_not_warn_about_rewrites_for_a_pinned_job(tmp_path, capsys):
    queue = tmp_path / 'q.txt'
    queue.write_text('[gpu=1] python a.py --device cuda:1\n')
    assert gpuq.main(['validate', str(queue)]) == 0
    out = capsys.readouterr().out
    assert 'verbatim' in out.lower()


def test_validate_ignores_the_staging_area(tmp_path, capsys):
    # the whole point of the staging area is that half written lines live there
    queue = tmp_path / 'q.txt'
    queue.write_text(f'python a.py\n{BAR}\n[exlc] half written --devi\n')
    assert gpuq.main(['validate', str(queue)]) == 0
    out = capsys.readouterr().out
    assert 'python a.py' in out
    assert 'staged' in out.lower()


def test_validate_skips_running_markers(tmp_path, capsys):
    queue = tmp_path / 'q.txt'
    queue.write_text(f'### RUNNING ON CUDA:0 ### python a.py\npython b.py\n{BAR}\n')
    assert gpuq.main(['validate', str(queue)]) == 0
    out = capsys.readouterr().out
    assert '1 job(s)' in out
    assert 'running' in out.lower()


def test_ensure_queue_file_creates_one_with_a_barrier(tmp_path):
    path = tmp_path / 'q.txt'
    assert gpuq.ensure_queue_file(path) is True
    doc = gpuq.parse_queue_document(path.read_text())
    assert doc.barrier is not None
    assert all(not line.strip() or line.strip().startswith('#') for line in doc.content)


def test_ensure_queue_file_never_touches_an_existing_one(tmp_path):
    path = tmp_path / 'q.txt'
    path.write_text('python a.py\n')
    assert gpuq.ensure_queue_file(path) is False
    assert path.read_text() == 'python a.py\n'


def test_validate_fails_loudly_on_a_bad_directive(tmp_path, capsys):
    queue = tmp_path / 'q.txt'
    queue.write_text('[exlc] python a.py\n')
    assert gpuq.main(['validate', str(queue)]) != 0
    assert 'exlc' in capsys.readouterr().err


def test_validate_reports_a_missing_queue_file(tmp_path, capsys):
    assert gpuq.main(['validate', str(tmp_path / 'nope.txt')]) != 0
    assert 'nope.txt' in capsys.readouterr().err


def test_kill_reports_when_nothing_is_running(tmp_path, capsys, monkeypatch):
    # one idle gpu 0 with no foreign procs, so nvidia-smi is never spawned
    monkeypatch.setattr(gpuq, 'nvidia_smi_probe', lambda *args, **kwargs: [gpuq.GpuState(index=0)])
    rc = gpuq.main(
        ['kill', 'cuda:0', '--log-dir', str(tmp_path), '--yes', '--floor-mib', '99999999']
    )
    assert rc != 0
    assert 'no gpuq job' in capsys.readouterr().err.lower()


def test_status_on_a_fresh_log_dir_is_not_an_error(tmp_path, capsys):
    assert gpuq.main(['status', '--log-dir', str(tmp_path)]) == 0


def test_status_shows_a_running_job(watcher_factory, capsys):
    w = watcher_factory('sleep 3\n')
    w.tick()
    capsys.readouterr()
    assert gpuq.main(['status', '--log-dir', str(w.log_dir)]) == 0
    out = capsys.readouterr().out
    assert str(w.state.running[0].pid) in out


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        gpuq.main([])


# ---------------------------------------------- no side effects before the lock


def test_constructing_a_watcher_writes_nothing(watcher_factory):
    # reattach must wait for the first tick, so a watcher that turns out to be
    # locked out never touches the shared ledger
    w = watcher_factory('sleep 3\n')
    w.tick()
    w.detach()
    before = len(ledger_events(w))

    w2 = gpuq.Watcher(
        queue_file=w.queue_path,
        log_dir=w.log_dir,
        gpu_probe=w.probe,
        settle_secs=0.0,
        cwd=w.cwd,
    )
    assert len(ledger_events(w2)) == before
    w2.tick()
    assert len(ledger_events(w2)) > before
    w2.shutdown_hard()


def test_a_locked_out_watcher_leaves_the_ledger_alone(watcher_factory):
    w = watcher_factory('sleep 3\n')
    w.tick()
    w.acquire_lock()
    before = len(ledger_events(w))

    w2 = gpuq.Watcher(
        queue_file=w.queue_path,
        log_dir=w.log_dir,
        gpu_probe=w.probe,
        settle_secs=0.0,
        cwd=w.cwd,
    )
    with pytest.raises(gpuq.WatcherLockError):
        w2.acquire_lock()
    assert len(ledger_events(w2)) == before


def test_kill_leaves_reaping_to_the_watcher(watcher_factory, monkeypatch):
    # kill must not rewrite state.json: a save built from a stale read can rewind
    # the queue cursor and replay work the watcher already ingested
    monkeypatch.setattr(gpuq, 'nvidia_smi_probe', lambda *args, **kwargs: [gpuq.GpuState(index=0)])
    w = watcher_factory('sleep 30\n')
    w.tick()
    jid = w.state.running[0].jid

    rc = gpuq.main(
        [
            'kill',
            'cuda:0',
            '--log-dir',
            str(w.log_dir),
            '--yes',
            '--floor-mib',
            '99999999',
            '--sigint-grace',
            '3',
            '--sigterm-grace',
            '3',
        ]
    )
    assert rc == 0
    assert [e for e in ledger_events(w) if e['event'] == 'killed']

    on_disk = gpuq.load_state(w.state_path)
    assert [r.jid for r in on_disk.running] == [jid]
    assert tick_until(w, lambda: not w.state.running, timeout=10)


# ------------------------------------------------------------------ pause flag
#
# A '# pause' comment anywhere above the barrier stops NEW launches. Everything
# else (reaping, marker rewriting, state, the ledger) carries on, and the
# jobs already running are left alone. Delete the line to resume.


@pytest.mark.parametrize(
    'line',
    [
        pytest.param('# pause', id='plain'),
        pytest.param('#pause', id='no_space'),
        pytest.param('##  PAUSE', id='upper_case'),
        pytest.param('# gpuq: pause', id='gpuq_prefix'),
        pytest.param('#   Pause  ', id='padded'),
    ],
)
def test_a_pause_line_marks_the_document_paused(line):
    assert gpuq.parse_queue_document(line + '\npython t.py\n').paused, line


@pytest.mark.parametrize(
    ('text', 'has_barrier'),
    [
        pytest.param('python t.py\n', False, id='no_pause_line'),
        # staging is a scratch area, the watcher never reads anything down there
        pytest.param('python t.py\n#### barrier ####\n# pause\n', False, id='below_barrier'),
        pytest.param('# paused this yesterday\n', False, id='mentions_paused'),
        pytest.param('# unpause when 15.4 lands\n', False, id='mentions_unpause'),
        pytest.param('# pauses\n', False, id='mentions_pauses'),
        # four or more hashes is the barrier syntax and has to keep winning
        pytest.param('#### pause ####\npython t.py\n', True, id='four_hash_barrier'),
    ],
)
def test_a_document_without_a_pause_line_is_not_paused(text, has_barrier):
    doc = gpuq.parse_queue_document(text)
    assert not doc.paused
    if has_barrier:
        assert doc.barrier is not None


def test_a_paused_queue_launches_nothing(watcher_factory):
    w = watcher_factory('# pause\nsleep 0.1\n')
    w.tick()
    assert w.state.running == []
    assert len(w.state.pending) == 1


def test_a_paused_watcher_still_reaps_the_jobs_already_running(watcher_factory):
    w = watcher_factory('sleep 0.1\n')
    w.tick()
    assert len(w.state.running) == 1
    w.queue_path.write_text('# pause\n' + w.queue_path.read_text())
    assert tick_until(w, lambda: not w.state.running), 'paused watcher never reaped'
    assert [e['event'] for e in ledger_events(w)].count('finished') == 1


def test_deleting_the_pause_line_resumes_launching(watcher_factory):
    w = watcher_factory('# pause\nsleep 0.1\n')
    w.tick()
    assert w.state.running == []
    w.queue_path.write_text('sleep 0.1\n')
    w.tick()
    assert len(w.state.running) == 1


def test_pause_survives_a_rewrite_that_adds_a_running_marker(watcher_factory):
    # running markers are rendered ABOVE the queued lines, so a pause typed at
    # the top gets pushed down. it has to keep working from wherever it lands.
    w = watcher_factory('sleep 5\n')
    w.tick()
    marker = w.queue_path.read_text().splitlines()[0]
    w.queue_path.write_text(f'# pause\n{marker}\nsleep 0.1\n')
    w.tick()

    text = w.queue_path.read_text()
    assert text.splitlines()[0].startswith('### RUNNING ON')  # pause got pushed down
    assert gpuq.parse_queue_document(text).paused
    assert len(w.state.running) == 1  # and the second job still did not start


def test_pausing_is_recorded_in_the_ledger_once_per_transition(watcher_factory):
    w = watcher_factory('# pause\nsleep 0.1\n')
    w.tick()
    w.tick()
    w.queue_path.write_text('sleep 0.1\n')
    w.tick()
    events = [e['event'] for e in ledger_events(w)]
    assert events.count('paused') == 1
    assert events.count('resumed') == 1


def test_pause_state_reaches_another_pane_through_state_json(watcher_factory):
    w = watcher_factory('# pause\nsleep 0.1\n')
    w.tick()
    assert gpuq.load_state(w.state_path).paused


# --------------------------------------------- job dependencies ([after=NAME])
#
# An analysis job must not run before the sweep it reads. A blocked job is
# SKIPPED rather than blocking the head, so the rest of the queue keeps moving
# and a dependency queued behind its dependent still resolves.


def mkdep(jid=1, name=None, after=(), exclusive=False, cmd='python t.py'):
    return gpuq.Job(
        jid=jid,
        name=name or f'job{jid}',
        cmd=cmd,
        exclusive=exclusive,
        pin_gpu=None,
        raw_line=cmd,
        lineno=jid,
        after=list(after),
    )


def test_after_directive_records_a_dependency():
    job = gpuq.parse_queue_line('[after=sweep] python a.py', lineno=1, jid=1)
    assert job.after == ['sweep']
    assert job.cmd == 'python a.py'


def test_after_can_be_repeated_for_several_dependencies():
    job = gpuq.parse_queue_line('[after=a, after=b] python c.py', lineno=1, jid=1)
    assert job.after == ['a', 'b']


def test_an_empty_after_value_is_a_hard_error():
    with pytest.raises(gpuq.QueueParseError):
        gpuq.parse_queue_line('[after=] python a.py', lineno=1, jid=1)


def test_a_job_waits_while_its_dependency_is_still_running():
    running = [mkrunning(jid=1, gpu=0)]  # named job1
    d = decide([mkdep(jid=2, after=['job1'])], running=running)
    assert d.action == 'wait'
    assert 'job1' in d.reason


def test_a_job_launches_once_its_dependency_has_finished_cleanly():
    d = decide([mkdep(jid=2, after=['sweep'])], finished={'sweep': 0})
    assert d.action == 'launch'


def test_a_blocked_job_is_skipped_so_the_queue_keeps_moving():
    # without the skip, an analysis job waiting on a 40 h sweep would idle the
    # whole box behind it
    pending = [mkdep(jid=1, name='analysis', after=['sweep']), mkdep(jid=2, name='other')]
    running = [
        gpuq.RunningJob(
            jid=9,
            name='sweep',
            gpu=0,
            pid=9,
            starttime=1,
            exclusive=False,
            log_path='/dev/null',
            cmd_as_run='x',
            started_at=0.0,
        )
    ]
    d = decide(pending, running=running)
    assert d.action == 'launch'
    assert d.job.name == 'other'


def test_a_job_whose_dependency_failed_is_errored_out_not_run():
    # analysis on a half-finished sweep produces a plausible table, not an error
    d = decide([mkdep(jid=2, name='analysis', after=['sweep'])], finished={'sweep': 1})
    assert d.action == 'error'
    assert 'sweep' in d.reason


def test_a_job_naming_a_dependency_that_does_not_exist_is_errored_out():
    # a typo must fail loudly rather than wait forever
    d = decide([mkdep(jid=2, after=['typo'])])
    assert d.action == 'error'
    assert 'typo' in d.reason


def test_a_dependency_queued_behind_its_dependent_still_resolves():
    pending = [mkdep(jid=1, name='analysis', after=['sweep']), mkdep(jid=2, name='sweep')]
    d = decide(pending)
    assert d.action == 'launch'
    assert d.job.name == 'sweep'


def test_a_blocked_job_does_not_overtake_an_exclusive_job():
    pending = [
        mkdep(jid=1, name='analysis', after=['sweep']),
        mkdep(jid=2, name='big', exclusive=True),
        mkdep(jid=3, name='small'),
    ]
    running = [
        gpuq.RunningJob(
            jid=9,
            name='sweep',
            gpu=0,
            pid=9,
            starttime=1,
            exclusive=False,
            log_path='/dev/null',
            cmd_as_run='x',
            started_at=0.0,
        )
    ]
    d = decide(pending, running=running)
    assert d.action == 'wait'  # 'big' is next in line and must drain first


def test_a_dependency_with_an_unknown_exit_code_holds_the_job():
    # an adopted job is not our child, so we never see its exit code. running
    # analysis on it is a guess, so the job holds and says so.
    status, reason = gpuq.dependency_status(
        mkdep(jid=2, after=['sweep']), running=[], pending=[], finished={'sweep': None}
    )
    assert status == 'hold'
    assert 'unknown' in reason.lower()


def test_a_dependent_job_runs_only_after_its_dependency_finishes(watcher_factory):
    w = watcher_factory('[name=first] sleep 0.6\n[after=first] echo second\n')
    w.tick()
    assert [r.name for r in w.state.running] == ['first']
    assert len(w.state.pending) == 1
    assert drain(w), 'queue never drained'
    started = [e['name'] for e in ledger_events(w) if e['event'] == 'started']
    assert started == ['first', 'job0002']


def test_a_dependent_job_is_dropped_when_its_dependency_fails(watcher_factory):
    w = watcher_factory('[name=first] exit 3\n[after=first, name=second] echo second\n')
    assert tick_until(w, lambda: not w.state.pending and not w.state.running)
    started = [e['name'] for e in ledger_events(w) if e['event'] == 'started']
    assert started == ['first']
    assert 'second' not in ' '.join(started)
    done = w.done_path.read_text()
    assert 'ERROR' in done and 'first' in done


def test_a_finished_job_is_recorded_by_name_for_later_dependents(watcher_factory):
    w = watcher_factory('[name=first] echo hi\n')
    assert drain(w)
    assert gpuq.load_state(w.state_path).finished == {'first': 0}


# ------------------------------------- pause + after on the cli and in the docs


def test_validate_says_the_queue_is_paused(tmp_path, capsys):
    queue = tmp_path / 'q.txt'
    queue.write_text('# pause\npython a.py\n')
    assert gpuq.main(['validate', str(queue)]) == 0
    assert 'paused' in capsys.readouterr().out.lower()


def test_validate_reports_a_jobs_dependencies(tmp_path, capsys):
    queue = tmp_path / 'q.txt'
    queue.write_text('[name=sweep] python a.py\n[after=sweep] python b.py\n')
    assert gpuq.main(['validate', str(queue)]) == 0
    out = capsys.readouterr().out
    assert 'after' in out.lower() and 'sweep' in out


def test_status_says_the_queue_is_paused(watcher_factory, capsys):
    w = watcher_factory('# pause\nsleep 0.1\n')
    w.tick()
    capsys.readouterr()
    assert gpuq.main(['status', '--log-dir', str(w.log_dir)]) == 0
    assert 'paused' in capsys.readouterr().out.lower()


# --------------------------------------------------- junk lines in the queue
#
# The queue file is live, so a stray paste or a redirect can drop something
# that is not a command into it. A stray paste can drop a monitor row into the
# queue, and it must never launch as a job. A line only launches if its first
# real token is something the shell could actually execute.

MONITOR_ROW = (
    '2601546 alice   0 Compute   0%    246MiB   2%     4%    911MiB /usr/bin/python train.py'
)


def test_a_pasted_gpu_monitor_row_is_not_launchable():
    problem = gpuq.launch_problem(MONITOR_ROW)
    assert problem is not None
    assert '2601546' in problem


@pytest.mark.parametrize(
    'cmd',
    [
        pytest.param('/bin/echo hello', id='absolute_path'),
        pytest.param('echo hello', id='program_on_path'),
        # 'cd' is a shell builtin, so which() will never find it
        pytest.param('cd /tmp && echo hi', id='cd_chain'),
        pytest.param('FOO=1 BAR=2 echo hi', id='env_assignment'),
        # shlex cannot split this, but the first token is still readable
        pytest.param("echo it's fine", id='unbalanced_quote'),
    ],
)
def test_a_runnable_command_is_launchable(cmd):
    assert gpuq.launch_problem(cmd) is None


@pytest.mark.parametrize(
    ('cmd', 'fragment'),
    [
        pytest.param(
            'definitely-not-a-real-program-xyz --flag',
            'definitely-not-a-real-program-xyz',
            id='missing_program',
        ),
        pytest.param('   ', 'the line is empty', id='empty'),
    ],
)
def test_an_unrunnable_command_is_not_launchable(cmd, fragment):
    problem = gpuq.launch_problem(cmd)
    assert problem is not None
    assert fragment in problem


def test_the_check_runs_even_while_an_exclusive_job_holds_the_box():
    # junk should not sit in the queue waiting for a gpu it can never use
    running = [mkrunning(jid=9, gpu=None, exclusive=True)]
    d = decide([mkdep(jid=1, name='junk', cmd=MONITOR_ROW)], running=running)
    assert d.action == 'error'


def test_a_junk_line_is_dropped_into_the_done_file_without_running(watcher_factory):
    w = watcher_factory(MONITOR_ROW + '\n/bin/echo real-job\n')
    assert tick_until(w, lambda: not w.state.pending and not w.state.running)
    started = [e['name'] for e in ledger_events(w) if e['event'] == 'started']
    assert len(started) == 1  # only the real job
    done = w.done_path.read_text()
    assert 'ERROR' in done and '2601546' in done


def test_validate_flags_a_line_that_could_never_launch(tmp_path, capsys):
    queue = tmp_path / 'q.txt'
    queue.write_text(MONITOR_ROW + '\n')
    assert gpuq.main(['validate', str(queue)]) == 0
    out = capsys.readouterr().out
    assert '2601546' in out and 'not' in out.lower()
