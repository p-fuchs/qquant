"""Thin, mockable wrapper over the ``vastai`` CLI.

Every interaction with vast.ai funnels through one injectable seam (``Runner``) so the
whole module is unit-testable with no network and no GPU, and the orchestration driver
(Spec 08) never shells out ad hoc. ``wait_until_running`` takes injected ``sleep``/
``now`` callables so polling is bounded and tests never sleep for real.

Import-time torch-free: only ``subprocess`` / ``json`` / ``dataclasses`` are used.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# The single mock seam. Default runs the real CLI; tests inject a fake.
Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True)


# Verified offer query (decision-log "vast.ai"): RTX 4090, 1 GPU, verified+rentable,
# >=1 direct port, CUDA >= 12.6, ordered cheapest-by-dlperf first.
DEFAULT_OFFER_QUERY: str = (
    "gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true "
    "direct_port_count>=1 cuda_max_good>=12.6"
)
DEFAULT_ORDER: str = "dlperf_usd-"
RUNNING: str = "running"
TERMINAL_BAD: frozenset[str] = frozenset({"exited", "offline", "unknown"})


@dataclass(frozen=True)
class Offer:
    ask_id: int
    dph_total: float  # $/hr (compute); used for select_cheapest
    gpu_name: str
    num_gpus: int
    cuda_max_good: float
    reliability: float
    raw: dict  # full --raw record, kept for forward-compat


@dataclass(frozen=True)
class Instance:
    id: int
    actual_status: str | None  # the field we gate on
    cur_state: str | None
    status_msg: str | None
    ssh_host: str | None
    ssh_port: int | None
    raw: dict


class VastError(RuntimeError):
    """Any nonzero rc / unparseable ``--raw`` output."""


class VastTimeout(VastError):
    """Deadline reached, instance still not running."""


class VastNonRunning(VastError):
    """``actual_status`` entered a terminal-bad state (exited/offline/unknown)."""


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_offer(rec: dict) -> Offer:
    return Offer(
        ask_id=int(rec["id"]),
        dph_total=float(rec.get("dph_total", 0.0)),
        gpu_name=str(rec.get("gpu_name", "")),
        num_gpus=int(rec.get("num_gpus", 0) or 0),
        cuda_max_good=float(rec.get("cuda_max_good", 0.0) or 0.0),
        # vast.ai exposes `reliability2` (newer) and `reliability`; prefer the former.
        reliability=float(rec.get("reliability2", rec.get("reliability", 0.0)) or 0.0),
        raw=rec,
    )


def _parse_instance(rec: dict) -> Instance:
    return Instance(
        id=int(rec["id"]),
        actual_status=rec.get("actual_status"),
        cur_state=rec.get("cur_state"),
        status_msg=rec.get("status_msg"),
        ssh_host=rec.get("ssh_host"),
        ssh_port=_as_int(rec.get("ssh_port")),
        raw=rec,
    )


class VastClient:
    """All methods build ``[binary, *args]`` and run via the injected ``runner``."""

    def __init__(
        self, runner: Runner = default_runner, *, binary: str = "vastai"
    ) -> None:
        self.runner = runner
        self.binary = binary

    # -- internal seam --

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        argv = [self.binary, *args]
        proc = self.runner(argv)
        if proc.returncode != 0:
            raise VastError(
                f"`{' '.join(argv)}` failed (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '').strip()}"
            )
        return proc

    def _run_json(self, *args: str) -> Any:
        proc = self._run(*args)
        try:
            return json.loads(proc.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            cmd = " ".join([self.binary, *args])
            raise VastError(
                f"`{cmd}` returned unparseable --raw output: {exc}"
            ) from exc

    # -- lifecycle --

    def search_offers(
        self,
        query: str = DEFAULT_OFFER_QUERY,
        *,
        order: str = DEFAULT_ORDER,
        limit: int = 32,
    ) -> list[Offer]:
        data = self._run_json(
            "search", "offers", query, "-o", order, "--limit", str(limit), "--raw"
        )
        if not isinstance(data, list):
            raise VastError("search offers --raw did not return a list")
        return [_parse_offer(rec) for rec in data]

    def create_instance(
        self,
        ask_id: int,
        *,
        image: str,
        disk_gb: int = 120,
        onstart_cmd: str | None = None,
        env: dict[str, str] | None = None,
        label: str | None = None,
        ssh: bool = True,
        direct: bool = True,
    ) -> int:
        args = [
            "create",
            "instance",
            str(ask_id),
            "--image",
            image,
            "--disk",
            str(disk_gb),
        ]
        if onstart_cmd is not None:
            args += ["--onstart-cmd", onstart_cmd]
        if env:
            args += ["--env", " ".join(f"-e {k}={v}" for k, v in env.items())]
        if label is not None:
            args += ["--label", label]
        if ssh:
            args.append("--ssh")
        if direct:
            args.append("--direct")
        args.append("--raw")
        data = self._run_json(*args)
        new_id = _as_int(data.get("new_contract")) if isinstance(data, dict) else None
        if new_id is None and isinstance(data, dict):
            new_id = _as_int(data.get("new_instance"))
        if new_id is None:
            raise VastError(f"create instance: no new_contract id in {data!r}")
        return new_id

    def show_instance(self, instance_id: int) -> Instance:
        data = self._run_json("show", "instance", str(instance_id), "--raw")
        if not isinstance(data, dict):
            raise VastError("show instance --raw did not return an object")
        return _parse_instance(data)

    def list_instances(self) -> list[Instance]:
        """All of the user's current instances (used for label-scoped orphan sweeps)."""
        data = self._run_json("show", "instances", "--raw")
        if not isinstance(data, list):
            raise VastError("show instances --raw did not return a list")
        return [_parse_instance(rec) for rec in data]

    def ssh_url(self, instance_id: int) -> str:
        return self._run("ssh-url", str(instance_id)).stdout.strip()

    def ssh_exec(
        self,
        instance_id: int,
        command: str,
        *,
        timeout_s: float | None = None,
        connect_timeout_s: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        url = self.ssh_url(instance_id)  # ssh://root@host:port
        host, port = _parse_ssh_url(url)
        ssh_argv = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            f"ConnectTimeout={connect_timeout_s}",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(port),
            f"root@{host}",
        ]
        # A wall-clock cap on the *remote* command (coreutils `timeout`) so a hung
        # download/build self-kills and the caller advances to teardown instead of
        # blocking forever on a billing instance. ssh ConnectTimeout only caps connect.
        if timeout_s is not None:
            ssh_argv.append(f"timeout {int(timeout_s)} sh -c {shlex.quote(command)}")
        else:
            ssh_argv.append(command)
        return self.runner(ssh_argv)

    def copy_to(self, instance_id: int, local_path: str, remote_path: str) -> None:
        self._run("copy", local_path, f"{instance_id}:{remote_path}")

    def copy_from(self, instance_id: int, remote_path: str, local_path: str) -> None:
        self._run("copy", f"{instance_id}:{remote_path}", local_path)

    def destroy(self, instance_id: int) -> None:
        """Idempotent: swallow a nonzero rc that means the instance is already gone.

        ``-y`` is mandatory: without it the CLI prompts for confirmation and this
        non-interactive subprocess would hang, leaking the instance (runaway cost).
        """
        argv = [self.binary, "destroy", "instance", str(instance_id), "-y"]
        proc = self.runner(argv)
        if proc.returncode == 0:
            return
        msg = (proc.stderr or proc.stdout or "").lower()
        if any(
            token in msg
            for token in ("not found", "no such", "already", "does not exist")
        ):
            return
        raise VastError(
            f"`{' '.join(argv)}` failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )


def _parse_ssh_url(url: str) -> tuple[str, int]:
    """``ssh://root@host:port`` -> ``(host, port)``."""
    rest = url.split("://", 1)[-1]
    if "@" in rest:
        rest = rest.split("@", 1)[1]
    host, _, port = rest.partition(":")
    return host, int(port or 22)


def select_cheapest(
    offers: list[Offer],
    *,
    max_dph: float,
    min_reliability: float = 0.95,
    min_cuda: float = 12.6,
) -> Offer:
    """Cheapest offer (min ``dph_total``) meeting price/reliability/CUDA gates.

    Raises ``VastError`` if none qualify.
    """
    eligible = [
        o
        for o in offers
        if o.dph_total <= max_dph
        and o.reliability >= min_reliability
        and o.cuda_max_good >= min_cuda
    ]
    if not eligible:
        raise VastError(
            f"no offer meets max_dph={max_dph}, min_reliability={min_reliability}, "
            f"min_cuda={min_cuda} (had {len(offers)} candidate offers)"
        )
    return min(eligible, key=lambda o: o.dph_total)


def wait_until_running(
    client: VastClient,
    instance_id: int,
    *,
    timeout_s: float = 900.0,
    poll_interval_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    max_consecutive_errors: int = 5,
) -> Instance:
    """Poll ``show_instance`` until running with an ``ssh_host``.

    Raises ``VastNonRunning`` if ``actual_status`` enters ``TERMINAL_BAD``, and
    ``VastTimeout`` at the deadline. A transient ``show_instance`` error (e.g. a 429/5xx
    blip) is tolerated up to ``max_consecutive_errors`` consecutive times before giving
    up, so one flaky poll never aborts a paid run. ``sleep``/``now`` are injected so
    polling is bounded and tests never sleep for real.
    """
    start = now()
    errors = 0
    while True:
        try:
            inst = client.show_instance(instance_id)
            errors = 0
        except VastError as exc:
            errors += 1
            if errors > max_consecutive_errors:
                raise VastError(
                    f"instance {instance_id}: {errors} consecutive show_instance "
                    f"errors; last: {exc}"
                ) from exc
            if now() - start >= timeout_s:
                raise VastTimeout(
                    f"instance {instance_id} not running after {timeout_s}s "
                    f"(last error: {exc})"
                ) from exc
            sleep(poll_interval_s)
            continue

        status = inst.actual_status
        if status == RUNNING and inst.ssh_host:
            return inst
        if status in TERMINAL_BAD:
            raise VastNonRunning(
                f"instance {instance_id} status={status!r}: {inst.status_msg}"
            )
        if now() - start >= timeout_s:
            raise VastTimeout(
                f"instance {instance_id} not running after {timeout_s}s "
                f"(last status={status!r})"
            )
        sleep(poll_interval_s)
