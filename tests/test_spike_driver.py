"""Unit tests for the spike driver.

A fake VastClient (no network/GPU) drives ``run_spike``; ``copy_from`` writes a canned
verdict payload to the local path so the driver can parse it. The central guarantee is
teardown: exactly one ``destroy`` on every exit path (and ``--keep-alive`` skips it).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from qquant.orchestrate.spike import (
    ALL_CHECKS,
    DEFAULT_LM_EVAL_CANDIDATES,
    GATING_CHECKS,
    QQUANT_IMAGE,
    SpikeVerdict,
    main,
    run_spike,
    verdict_from_payload,
)
from qquant.orchestrate.vastai import (
    DEFAULT_OFFER_QUERY,
    Instance,
    Offer,
    VastNonRunning,
)


def _payload(*, lm_eval_ref="abc123sha", awq=True, fail=None):
    checks = {k: True for k in GATING_CHECKS}
    if fail is not None:
        checks[fail] = False
    return {
        "verdict": "GO" if all(checks.values()) else "NO-GO",
        "lm_eval_ref": lm_eval_ref,
        "lm_eval_version": "0.4.12",
        "transformers_version": "5.10.1",
        "torch_version": "2.12.1+cu126",
        "torch_cuda": "12.6",
        "nvcc_cuda": "12.6",
        "checks": dict(checks, awq_official_loads=awq),
        "awq_official_loads": awq,
        "notes": [],
    }


class FakeVastClient:
    def __init__(
        self,
        *,
        status="running",
        ssh_host="1.2.3.4",
        new_id=4242,
        payloads=None,
        ssh_exec_raises=None,
        ssh_exec_rc=0,
        create_raises=None,
        instances=None,
    ):
        self._status = status
        self._ssh_host = ssh_host
        self._new_id = new_id
        self._payloads = list(payloads or [])
        self._ssh_exec_raises = ssh_exec_raises
        self._ssh_exec_rc = ssh_exec_rc
        self._create_raises = create_raises
        self._instances = list(instances or [])
        self.search_calls = 0
        self.create_ask_id = None
        self.create_kwargs = None
        self.copy_to_calls: list[tuple] = []
        self.copy_from_calls: list[tuple] = []
        self.ssh_exec_calls: list[str] = []
        self.destroy_calls: list[int] = []

    def search_offers(self, query=DEFAULT_OFFER_QUERY, *, order=None, limit=32):
        self.search_calls += 1
        return [
            Offer(
                ask_id=222,
                dph_total=0.30,
                gpu_name="RTX 4090",
                num_gpus=1,
                cuda_max_good=12.8,
                reliability=0.99,
                raw={},
            )
        ]

    def create_instance(self, ask_id, **kwargs):
        self.create_ask_id = ask_id
        self.create_kwargs = kwargs
        if self._create_raises is not None:
            raise self._create_raises
        return self._new_id

    def list_instances(self):
        return list(self._instances)

    def show_instance(self, instance_id):
        return Instance(
            id=instance_id,
            actual_status=self._status,
            cur_state=self._status,
            status_msg="boom",
            ssh_host=self._ssh_host,
            ssh_port=22,
            raw={},
        )

    def ssh_url(self, instance_id):
        return f"ssh://root@{self._ssh_host}:22"

    def ssh_exec(self, instance_id, command, *, timeout_s=None):
        self.ssh_exec_calls.append(command)
        if self._ssh_exec_raises is not None:
            raise self._ssh_exec_raises
        return subprocess.CompletedProcess([command], self._ssh_exec_rc, "", "")

    def copy_to(self, instance_id, local, remote):
        self.copy_to_calls.append((local, remote))

    def copy_from(self, instance_id, remote, local):
        self.copy_from_calls.append((remote, local))
        if not self._payloads:
            raise FileNotFoundError("no remote verdict")  # bootstrap never wrote one
        payload = self._payloads.pop(0)
        Path(local).write_text(json.dumps(payload))

    def destroy(self, instance_id):
        self.destroy_calls.append(instance_id)


# --------------------------------------------------------------------------------------
# happy path + candidate iteration
# --------------------------------------------------------------------------------------


def test_run_spike_happy_go(tmp_path):
    client = FakeVastClient(payloads=[_payload()])
    out = tmp_path / "spike.json"

    verdict = run_spike(client, image="img@sha", out_path=str(out))

    assert verdict.verdict == "GO"
    assert verdict.lm_eval_ref == "abc123sha"
    assert client.ssh_exec_calls[0] == "mkdir -p /workspace"  # /workspace created first
    candidate_runs = [c for c in client.ssh_exec_calls if "spike_remote.py" in c]
    assert len(candidate_runs) == 1  # first candidate already GO
    assert client.destroy_calls == [4242]
    assert client.create_ask_id == 222
    assert client.create_kwargs["disk_gb"] == 120
    # dead-man's switch installed via onstart_cmd (curl REST DELETE), runtime-bounded.
    onstart = client.create_kwargs["onstart_cmd"]
    assert "1800" in onstart and "/instances/" in onstart and "DELETE" in onstart
    assert "env" in client.create_kwargs
    # verdict persisted to out_path
    saved = json.loads(out.read_text())
    assert saved["verdict"] == "GO"


def test_run_spike_tries_candidates_until_go(tmp_path):
    # first candidate NO-GO (a gating check fails), second candidate GO.
    client = FakeVastClient(
        payloads=[_payload(fail="bnb_loads"), _payload(lm_eval_ref="0.4.12")]
    )
    verdict = run_spike(
        client,
        image="img",
        out_path=str(tmp_path / "s.json"),
        candidates=("main", "0.4.12"),
    )
    assert verdict.verdict == "GO"
    assert verdict.lm_eval_ref == "0.4.12"
    candidate_runs = [c for c in client.ssh_exec_calls if "spike_remote.py" in c]
    assert len(candidate_runs) == 2


def test_all_candidates_fail_bootstrap_is_nogo(tmp_path):
    # ssh_exec returns nonzero rc => bootstrap failed; never copy a verdict back.
    client = FakeVastClient(ssh_exec_rc=1, payloads=[])
    verdict = run_spike(
        client,
        image="img",
        out_path=str(tmp_path / "s.json"),
        candidates=("main", "0.4.12"),
    )
    assert verdict.verdict == "NO-GO"
    # copy_from is attempted per candidate (verdict file is the source of truth) but the
    # remote never wrote one, so it's recorded as a bootstrap failure.
    assert len(client.copy_from_calls) == 2
    assert verdict.notes  # explains why
    assert client.destroy_calls == [4242]


def test_nonzero_remote_with_verdict_is_used_not_discarded(tmp_path):
    # spike_remote exits 1 on NO-GO but still writes a full verdict; run_spike must READ
    # the verdict file (source of truth) rather than treat rc!=0 as a bootstrap failure.
    client = FakeVastClient(ssh_exec_rc=1, payloads=[_payload(fail="bnb_loads")])
    verdict = run_spike(
        client, image="img", out_path=str(tmp_path / "s.json"), candidates=("main",)
    )
    assert verdict.verdict == "NO-GO"
    assert verdict.checks["bnb_loads"] is False
    assert verdict.checks["nvcc_matches_torch"] is True  # real per-check data preserved
    assert client.copy_from_calls  # verdict was copied back despite rc != 0


# --------------------------------------------------------------------------------------
# teardown guarantee (Done-when #5)
# --------------------------------------------------------------------------------------


def test_destroys_when_ssh_exec_raises(tmp_path):
    client = FakeVastClient(ssh_exec_raises=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        run_spike(client, image="img", out_path=str(tmp_path / "s.json"))
    assert client.destroy_calls == [4242]
    assert client.ssh_exec_calls  # reached the remote step (mkdir) before raising


def test_destroys_when_wait_raises(tmp_path):
    client = FakeVastClient(status="exited")
    with pytest.raises(VastNonRunning):
        run_spike(client, image="img", out_path=str(tmp_path / "s.json"))
    assert client.destroy_calls == [4242]


def test_keep_alive_skips_destroy(tmp_path):
    client = FakeVastClient(payloads=[_payload()])
    run_spike(client, image="img", out_path=str(tmp_path / "s.json"), keep_alive=True)
    assert client.destroy_calls == []


def test_create_parse_failure_sweeps_orphan_by_label(tmp_path):
    # create raised AFTER vast already made the instance (id unknown): the label-scoped
    # orphan sweep must still destroy it so we don't leak a billing instance.
    from qquant.orchestrate.vastai import Instance, VastError

    orphan = Instance(
        id=777,
        actual_status="running",
        cur_state="running",
        status_msg=None,
        ssh_host="h",
        ssh_port=22,
        raw={"id": 777, "label": "qquant-spike"},
    )
    client = FakeVastClient(
        create_raises=VastError("no id in --raw"), instances=[orphan]
    )
    with pytest.raises(VastError):
        run_spike(client, image="img", out_path=str(tmp_path / "s.json"))
    assert client.destroy_calls == [777]


# --------------------------------------------------------------------------------------
# verdict shape + constants (Done-when #6, #7)
# --------------------------------------------------------------------------------------


def test_verdict_from_payload_shape():
    verdict = verdict_from_payload(_payload())
    assert isinstance(verdict, SpikeVerdict)
    assert verdict.verdict == "GO"
    assert verdict.lm_eval_ref == "abc123sha"
    assert verdict.lm_eval_version == "0.4.12"
    assert verdict.transformers_version == "5.10.1"
    assert verdict.torch_version == "2.12.1+cu126"
    assert verdict.torch_cuda == "12.6"
    assert verdict.nvcc_cuda == "12.6"
    assert verdict.awq_official_loads is True
    assert set(GATING_CHECKS).issubset(verdict.checks)
    assert set(ALL_CHECKS).issubset(verdict.checks)
    assert verdict.notes == []


def test_verdict_from_payload_nogo_when_gating_fails():
    verdict = verdict_from_payload(_payload(fail="gptq_loads"))
    assert verdict.verdict == "NO-GO"
    # a failing non-gating check does NOT block GO
    ok_but_no_awq = verdict_from_payload(_payload(awq=False))
    assert ok_but_no_awq.verdict == "GO"
    assert ok_but_no_awq.awq_official_loads is False


def test_constants():
    assert QQUANT_IMAGE == "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04"
    assert DEFAULT_LM_EVAL_CANDIDATES == ("main", "0.4.12")


def test_main_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
