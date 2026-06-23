"""Unit tests for the vast.ai lifecycle wrapper.

No network, no GPU, no real sleep: every CLI call funnels through an injected fake
``Runner`` and ``wait_until_running`` takes injected ``sleep``/``now`` callables.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from qquant.orchestrate.vastai import (
    DEFAULT_OFFER_QUERY,
    DEFAULT_ORDER,
    Instance,
    Offer,
    VastClient,
    VastError,
    VastNonRunning,
    VastTimeout,
    select_cheapest,
    wait_until_running,
)


class FakeRunner:
    """Records argv and returns scripted ``CompletedProcess`` responses.

    ``responses`` is a list of ``(returncode, stdout, stderr)`` consumed FIFO; once
    exhausted, ``default`` is returned for every further call.
    """

    def __init__(self, responses=None, *, default=(0, "", "")):
        self._responses = list(responses or [])
        self._default = default
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        rc, out, err = self._responses.pop(0) if self._responses else self._default
        return subprocess.CompletedProcess(argv, rc, out, err)


# --------------------------------------------------------------------------------------
# fixtures / canned data
# --------------------------------------------------------------------------------------

_OFFERS = [
    {
        "id": 111,
        "dph_total": 0.50,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "cuda_max_good": 12.8,
        "reliability2": 0.99,
    },
    {
        "id": 222,
        "dph_total": 0.30,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "cuda_max_good": 12.8,
        "reliability2": 0.97,
    },
    {
        "id": 333,
        "dph_total": 0.20,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "cuda_max_good": 12.4,
        "reliability2": 0.99,
    },  # cuda too low
    {
        "id": 444,
        "dph_total": 0.25,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "cuda_max_good": 12.8,
        "reliability2": 0.80,
    },  # reliability too low
]


def _inst(status, *, ssh_host=None, ssh_port=None, msg=None):
    return json.dumps(
        {
            "id": 98765,
            "actual_status": status,
            "cur_state": status,
            "status_msg": msg,
            "ssh_host": ssh_host,
            "ssh_port": ssh_port,
        }
    )


def _offers_from(client):
    return client.search_offers()


# --------------------------------------------------------------------------------------
# search_offers + select_cheapest
# --------------------------------------------------------------------------------------


def test_search_offers_argv_and_parsing():
    runner = FakeRunner([(0, json.dumps(_OFFERS), "")])
    client = VastClient(runner=runner)

    offers = client.search_offers()

    assert runner.calls[0] == [
        "vastai",
        "search",
        "offers",
        DEFAULT_OFFER_QUERY,
        "-o",
        DEFAULT_ORDER,
        "--limit",
        "32",
        "--raw",
    ]
    assert len(offers) == 4
    first = offers[0]
    assert isinstance(first, Offer)
    assert first.ask_id == 111
    assert first.dph_total == 0.50
    assert first.gpu_name == "RTX 4090"
    assert first.num_gpus == 1
    assert first.cuda_max_good == 12.8
    assert first.reliability == 0.99
    assert first.raw["id"] == 111


def test_select_cheapest_picks_min_dph_eligible():
    runner = FakeRunner([(0, json.dumps(_OFFERS), "")])
    offers = _offers_from(VastClient(runner=runner))

    chosen = select_cheapest(offers, max_dph=0.60)

    # 333 (cheapest) is excluded by cuda gate; 444 by reliability gate -> 222 wins.
    assert chosen.ask_id == 222
    assert chosen.dph_total == 0.30


def test_select_cheapest_raises_when_none_affordable():
    runner = FakeRunner([(0, json.dumps(_OFFERS), "")])
    offers = _offers_from(VastClient(runner=runner))

    with pytest.raises(VastError):
        select_cheapest(offers, max_dph=0.10)


# --------------------------------------------------------------------------------------
# wait_until_running (bounded, injected clock — never sleeps for real)
# --------------------------------------------------------------------------------------


def test_wait_until_running_returns_when_running():
    runner = FakeRunner(
        [
            (0, _inst("loading"), ""),
            (0, _inst("loading"), ""),
            (0, _inst("running", ssh_host="1.2.3.4", ssh_port=12345), ""),
        ]
    )
    client = VastClient(runner=runner)
    slept: list[float] = []
    clock = [0.0]

    inst = wait_until_running(
        client,
        98765,
        timeout_s=900.0,
        poll_interval_s=15.0,
        sleep=lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)),
        now=lambda: clock[0],
    )

    assert isinstance(inst, Instance)
    assert inst.actual_status == "running"
    assert inst.ssh_host == "1.2.3.4"
    assert inst.ssh_port == 12345
    assert slept == [15.0, 15.0]  # injected sleep used; no real sleeping


def test_wait_until_running_requires_ssh_host():
    # running but no ssh_host yet -> keep polling until host appears.
    runner = FakeRunner(
        [
            (0, _inst("running"), ""),  # running but ssh_host is None
            (0, _inst("running", ssh_host="5.6.7.8", ssh_port=22), ""),
        ]
    )
    client = VastClient(runner=runner)
    clock = [0.0]

    inst = wait_until_running(
        client,
        98765,
        timeout_s=900.0,
        poll_interval_s=10.0,
        sleep=lambda s: clock.__setitem__(0, clock[0] + s),
        now=lambda: clock[0],
    )
    assert inst.ssh_host == "5.6.7.8"


@pytest.mark.parametrize("bad", ["exited", "offline", "unknown"])
def test_wait_until_running_terminal_bad(bad):
    runner = FakeRunner([(0, _inst(bad, msg="boom"), "")])
    client = VastClient(runner=runner)

    with pytest.raises(VastNonRunning):
        wait_until_running(
            client,
            98765,
            timeout_s=900.0,
            poll_interval_s=15.0,
            sleep=lambda s: None,
            now=lambda: 0.0,
        )


def test_wait_until_running_tolerates_transient_errors():
    # two transient poll errors then running -> must not abort a paid run.
    runner = FakeRunner(
        [
            (1, "", "429 too many requests"),
            (1, "", "5xx upstream"),
            (0, _inst("running", ssh_host="h", ssh_port=22), ""),
        ]
    )
    client = VastClient(runner=runner)
    clock = [0.0]

    inst = wait_until_running(
        client,
        98765,
        timeout_s=900.0,
        poll_interval_s=5.0,
        sleep=lambda s: clock.__setitem__(0, clock[0] + s),
        now=lambda: clock[0],
    )
    assert inst.actual_status == "running"


def test_wait_until_running_gives_up_after_too_many_errors():
    runner = FakeRunner(default=(1, "", "boom"))  # every poll errors
    client = VastClient(runner=runner)

    with pytest.raises(VastError):
        wait_until_running(
            client,
            98765,
            timeout_s=900.0,
            poll_interval_s=1.0,
            sleep=lambda s: None,
            now=lambda: 0.0,
            max_consecutive_errors=3,
        )


def test_wait_until_running_times_out():
    runner = FakeRunner(default=(0, _inst("loading"), ""))  # never reaches running
    client = VastClient(runner=runner)
    clock = [0.0]

    with pytest.raises(VastTimeout):
        wait_until_running(
            client,
            98765,
            timeout_s=30.0,
            poll_interval_s=15.0,
            sleep=lambda s: clock.__setitem__(0, clock[0] + s),
            now=lambda: clock[0],
        )


# --------------------------------------------------------------------------------------
# argv construction for every VastClient method (Done-when #4)
# --------------------------------------------------------------------------------------


def test_create_instance_argv_and_id():
    runner = FakeRunner([(0, json.dumps({"success": True, "new_contract": 98765}), "")])
    client = VastClient(runner=runner)

    new_id = client.create_instance(
        222, image="img@sha256:abc", disk_gb=120, onstart_cmd="setsid sh -c 'sleep 1'"
    )

    assert new_id == 98765
    assert runner.calls[0] == [
        "vastai",
        "create",
        "instance",
        "222",
        "--image",
        "img@sha256:abc",
        "--disk",
        "120",
        "--onstart-cmd",
        "setsid sh -c 'sleep 1'",
        "--ssh",
        "--direct",
        "--raw",
    ]


def test_create_instance_with_env_and_label():
    runner = FakeRunner([(0, json.dumps({"success": True, "new_contract": 1}), "")])
    client = VastClient(runner=runner)

    client.create_instance(
        222, image="img", env={"VAST_API_KEY": "k", "HF_TOKEN": "t"}, label="spike"
    )

    argv = runner.calls[0]
    assert "--env" in argv
    assert argv[argv.index("--env") + 1] == "-e VAST_API_KEY=k -e HF_TOKEN=t"
    assert "--label" in argv
    assert argv[argv.index("--label") + 1] == "spike"


def test_show_instance_argv_and_parse():
    runner = FakeRunner([(0, _inst("running", ssh_host="h", ssh_port=22), "")])
    client = VastClient(runner=runner)

    inst = client.show_instance(98765)

    assert runner.calls[0] == ["vastai", "show", "instance", "98765", "--raw"]
    assert inst.id == 98765
    assert inst.actual_status == "running"
    assert inst.ssh_host == "h"
    assert inst.ssh_port == 22


def test_list_instances_argv_and_parse():
    payload = json.dumps(
        [
            {
                "id": 1,
                "actual_status": "running",
                "label": "qquant-spike",
                "ssh_host": "h",
            },
            {"id": 2, "actual_status": "exited", "label": "other"},
        ]
    )
    runner = FakeRunner([(0, payload, "")])
    client = VastClient(runner=runner)

    insts = client.list_instances()

    assert runner.calls[0] == ["vastai", "show", "instances", "--raw"]
    assert [i.id for i in insts] == [1, 2]
    assert insts[0].raw["label"] == "qquant-spike"


def test_ssh_exec_wraps_remote_timeout():
    # first runner call resolves ssh-url; second is the ssh invocation.
    runner = FakeRunner([(0, "ssh://root@host:2222\n", ""), (0, "ok", "")])
    client = VastClient(runner=runner)

    client.ssh_exec(98765, "echo hi", timeout_s=60)

    ssh_argv = runner.calls[1]
    assert ssh_argv[0] == "ssh"
    assert "ConnectTimeout=30" in ssh_argv
    assert "ServerAliveInterval=30" in ssh_argv
    assert ssh_argv[-3:-1] == ["-p", "2222"] or "-p" in ssh_argv
    assert ssh_argv[-2] == "root@host"
    # remote command is wrapped in coreutils `timeout` so a hang self-kills.
    assert ssh_argv[-1] == "timeout 60 sh -c 'echo hi'"


def test_ssh_url_argv_and_value():
    runner = FakeRunner([(0, "ssh://root@1.2.3.4:12345\n", "")])
    client = VastClient(runner=runner)

    url = client.ssh_url(98765)

    assert runner.calls[0] == ["vastai", "ssh-url", "98765"]
    assert url == "ssh://root@1.2.3.4:12345"


def test_copy_to_argv():
    runner = FakeRunner([(0, "", "")])
    client = VastClient(runner=runner)

    client.copy_to(98765, "local/spike_remote.py", "/workspace/spike_remote.py")

    assert runner.calls[0] == [
        "vastai",
        "copy",
        "local/spike_remote.py",
        "98765:/workspace/spike_remote.py",
    ]


def test_copy_from_argv():
    runner = FakeRunner([(0, "", "")])
    client = VastClient(runner=runner)

    client.copy_from(98765, "/workspace/spike_verdict.json", "out/spike.json")

    assert runner.calls[0] == [
        "vastai",
        "copy",
        "98765:/workspace/spike_verdict.json",
        "out/spike.json",
    ]


def test_copy_uses_identity_when_set():
    runner = FakeRunner([(0, "", ""), (0, "", "")])
    client = VastClient(runner=runner, ssh_identity="ssh_keys/vastai")

    client.copy_to(98765, "a.py", "/workspace/a.py")
    client.copy_from(98765, "/workspace/v.json", "out.json")

    assert runner.calls[0] == [
        "vastai",
        "copy",
        "-i",
        "ssh_keys/vastai",
        "a.py",
        "98765:/workspace/a.py",
    ]
    assert runner.calls[1] == [
        "vastai",
        "copy",
        "-i",
        "ssh_keys/vastai",
        "98765:/workspace/v.json",
        "out.json",
    ]


def test_ssh_exec_uses_identity_when_set():
    runner = FakeRunner([(0, "ssh://root@h:22\n", ""), (0, "ok", "")])
    client = VastClient(runner=runner, ssh_identity="ssh_keys/vastai")

    client.ssh_exec(98765, "echo hi", timeout_s=10)

    ssh_argv = runner.calls[1]
    assert "-i" in ssh_argv
    assert ssh_argv[ssh_argv.index("-i") + 1] == "ssh_keys/vastai"
    assert "IdentitiesOnly=yes" in ssh_argv


def test_destroy_argv():
    runner = FakeRunner([(0, "destroyed", "")])
    client = VastClient(runner=runner)

    client.destroy(98765)

    # -y is mandatory: non-interactive teardown must never block on a prompt.
    assert runner.calls[0] == ["vastai", "destroy", "instance", "98765", "-y"]


def test_destroy_is_idempotent_when_already_gone():
    runner = FakeRunner([(1, "", "instance 98765 not found")])
    client = VastClient(runner=runner)

    # Already-gone is swallowed (no raise): destroy must be idempotent.
    client.destroy(98765)


# --------------------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------------------


def test_nonzero_rc_raises_vast_error():
    runner = FakeRunner([(1, "", "auth failed")])
    client = VastClient(runner=runner)

    with pytest.raises(VastError):
        client.search_offers()


def test_unparseable_raw_raises_vast_error():
    runner = FakeRunner([(0, "this is not json", "")])
    client = VastClient(runner=runner)

    with pytest.raises(VastError):
        client.search_offers()
