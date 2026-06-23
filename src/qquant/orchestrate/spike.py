"""Go/no-go spike driver (the ``qquant-spike`` CLI).

Sequences the vast.ai wrapper to stand up one RTX 4090, run the on-box checks
(``scripts/spike/spike_remote.py``) under each candidate lm-eval ref, copy the verdict
back, and **guarantee teardown on every exit path**. Import-time torch-free: the GPU
work lives only in the remote script.

The verdict (GO/NO-GO + the resolved lm-eval ref) gates Specs 04-10.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from qquant.orchestrate.vastai import (
    DEFAULT_OFFER_QUERY,
    VastClient,
    select_cheapest,
    wait_until_running,
)

# Decision-log constant: CUDA 12.6 *devel* image (nvcc present, matches torch cu126).
# The spike validates the exact tag / SSH / sha256 digest on first pull.
QQUANT_IMAGE: str = "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04"

# Pinned model revisions (decision-log "Pinned model revisions").
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
GPTQ_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
GPTQ_REV = "e9c932ac1893a49ae0fc497ad6e1e86e2e39af20"
AWQ_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
AWQ_REV = "b25037543e9394b818fdfca67ab2a00ecc7dd641"

# Candidate refs tried in order; first to pass the gating checks wins. Fallback is last.
DEFAULT_LM_EVAL_CANDIDATES: tuple[str, ...] = ("main", "0.4.12")

# RTX 4090 on-demand ceiling ($/hr); guards against an over-priced offer.
DEFAULT_MAX_DPH: float = 0.80

# Gating checks (all must pass for GO) and the single non-gating contingency check.
GATING_CHECKS: tuple[str, ...] = (
    "nvcc_matches_torch",
    "mmlu_subjects_match",
    "chat_template_renders",
    "mmlu_subtask_scored",
    "generate_until_ok",
    "bnb_loads",
    "gptq_loads",
)
NONGATING_CHECKS: tuple[str, ...] = ("awq_official_loads",)
ALL_CHECKS: tuple[str, ...] = GATING_CHECKS + NONGATING_CHECKS

# Remote paths on the box. The bare CUDA-devel image has no Python/torch; the bootstrap
# uses uv to build a venv here, so the spike script runs under that interpreter (not the
# bare ``python``).
REMOTE_SCRIPT = "/workspace/spike_remote.py"
REMOTE_BOOTSTRAP = "/workspace/bootstrap.sh"
REMOTE_VERDICT = "/workspace/spike_verdict.json"
REMOTE_PYTHON = "/workspace/venv/bin/python"


@dataclass(frozen=True)
class SpikeVerdict:
    verdict: str  # "GO" | "NO-GO"
    lm_eval_ref: str  # resolved git SHA, or "0.4.12" (fallback pin)
    lm_eval_version: str
    transformers_version: str
    torch_version: str
    torch_cuda: str  # e.g. "12.6"
    nvcc_cuda: str  # e.g. "12.6"
    checks: dict[str, bool]
    awq_official_loads: bool  # NON-gating: flips variants.yaml awq-official.enabled
    notes: list[str]
    raw: dict  # full remote verdict json


def verdict_from_payload(raw: dict) -> SpikeVerdict:
    """Build a ``SpikeVerdict`` from a remote payload.

    The GO/NO-GO decision is recomputed locally from the gating checks (authoritative)
    rather than trusting the payload's own ``verdict`` string.
    """
    checks = {k: bool(v) for k, v in dict(raw.get("checks", {})).items()}
    for key in ALL_CHECKS:
        checks.setdefault(key, False)
    awq = bool(raw.get("awq_official_loads", checks.get("awq_official_loads", False)))
    is_go = all(checks.get(k, False) for k in GATING_CHECKS)
    return SpikeVerdict(
        verdict="GO" if is_go else "NO-GO",
        lm_eval_ref=str(raw.get("lm_eval_ref", "")),
        lm_eval_version=str(raw.get("lm_eval_version", "")),
        transformers_version=str(raw.get("transformers_version", "")),
        torch_version=str(raw.get("torch_version", "")),
        torch_cuda=str(raw.get("torch_cuda", "")),
        nvcc_cuda=str(raw.get("nvcc_cuda", "")),
        checks=checks,
        awq_official_loads=awq,
        notes=list(raw.get("notes", [])),
        raw=raw,
    )


def _nogo(notes: list[str]) -> SpikeVerdict:
    return SpikeVerdict(
        verdict="NO-GO",
        lm_eval_ref="",
        lm_eval_version="",
        transformers_version="",
        torch_version="",
        torch_cuda="",
        nvcc_cuda="",
        checks={k: False for k in ALL_CHECKS},
        awq_official_loads=False,
        notes=notes,
        raw={},
    )


def _repo_path(rel: str) -> Path:
    return Path(__file__).resolve().parents[3] / rel


def _deadman_onstart(max_runtime_s: int) -> str:
    """On-instance dead-man's switch (defense-in-depth backstop).

    The bare CUDA image has no python/pip, so this uses the vast.ai REST API via curl
    to self-destroy after ``max_runtime_s``. It runs at onstart, where the injected
    ``--env`` vars (``$VAST_API_KEY``, ``$CONTAINER_ID``) ARE present. BACKSTOP only:
    the primary guarantee is the local finally/atexit/signal teardown, and the operator
    account spend limit is the ultimate cap. The key is read from env (never on disk).
    NOTE: ``$CONTAINER_ID`` is assumed to be the numeric instance id — verify on the box
    (deadman logs to /workspace/deadman.log).
    """
    api = "https://console.vast.ai/api/v0/instances"
    return (
        # Bare CUDA images have no /workspace (a vast pytorch-template convention); make
        # it so the deadman log + the copied scripts have a home.
        "mkdir -p /workspace; "
        "command -v curl >/dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq curl ca-certificates) "
        ">/dev/null 2>&1; "
        f"setsid sh -c 'sleep {max_runtime_s}; "
        f'curl -s -X DELETE "{api}/$CONTAINER_ID/" '
        '-H "Authorization: Bearer $VAST_API_KEY"\' '
        ">/workspace/deadman.log 2>&1 &"
    )


def _remote_command(ref: str) -> str:
    # vast.ai ssh sessions don't inherit the image's PATH, so make nvcc (CUDA toolchain)
    # resolvable for both spike_remote's nvcc check and gptqmodel's sm_89 JIT build.
    # The diagnostic preamble (-> stderr, captured into notes on failure) makes a remote
    # failure self-explaining (PATH, tool presence, what landed in /workspace).
    diag = (
        "{ echo =DIAG=; uname -sm; echo PATH=$PATH; "
        "command -v timeout bash uv curl git nvcc; "
        "ls -la /workspace; echo =END-DIAG=; } >&2; "
    )
    return (
        "export PATH=/usr/local/cuda/bin:$PATH; "
        + diag
        # Remove any prior verdict so a stale one can't be misattributed to this ref.
        + f"rm -f {REMOTE_VERDICT}; "
        + f"bash {REMOTE_BOOTSTRAP} {ref} && "
        + f"{REMOTE_PYTHON} {REMOTE_SCRIPT} --lm-eval-ref {ref} "
        + f"--model {MODEL} --revision {MODEL_REV} "
        + f"--gptq-model {GPTQ_MODEL} --gptq-revision {GPTQ_REV} "
        + f"--awq-model {AWQ_MODEL} --awq-revision {AWQ_REV} "
        + f"--out {REMOTE_VERDICT}"
    )


def _write_verdict(out_path: str, verdict: SpikeVerdict) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(verdict), indent=2, sort_keys=True))


class _Teardown:
    """Always destroys the instance: finally + atexit + SIGINT/SIGTERM, with retries.

    Armed BEFORE the instance is created (``instance_id`` is assigned afterwards) so the
    create->arm window can't leak. If ``instance_id`` is still ``None`` on teardown
    (create raised, or a signal mid-create), it falls back to a label-scoped orphan
    sweep. ``_done`` flips only after a *successful* destroy, so a transient API error
    retries instead of silently leaking a billing instance.
    """

    def __init__(
        self,
        client: VastClient,
        instance_id: int | None,
        *,
        enabled: bool = True,
        label: str | None = None,
        retries: int = 3,
        retry_sleep_s: float = 2.0,
    ):
        self.client = client
        self.instance_id = instance_id
        self.enabled = enabled
        self.label = label
        self.retries = retries
        self.retry_sleep_s = retry_sleep_s
        self._done = False
        self._prev: dict = {}

    def _destroy_once(self) -> None:
        if self._done or not self.enabled:
            return
        if self.instance_id is None:
            self._sweep_orphans()
            self._done = True
            return
        for attempt in range(1, self.retries + 1):
            try:
                self.client.destroy(self.instance_id)
                self._done = True  # only after a successful destroy
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == self.retries:
                    print(
                        f"WARNING: failed to destroy instance {self.instance_id} after "
                        f"{self.retries} attempts ({exc}); destroy it manually or rely "
                        f"on the account spend limit",
                        file=sys.stderr,
                    )
                    return
                time.sleep(self.retry_sleep_s)

    def _sweep_orphans(self) -> None:
        """Created-but-id-unknown recovery: destroy any instance carrying our label."""
        if not self.label:
            return
        try:
            orphans = [
                i
                for i in self.client.list_instances()
                if i.raw.get("label") == self.label
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: orphan sweep failed: {exc}", file=sys.stderr)
            return
        for inst in orphans:
            try:
                self.client.destroy(inst.id)
                print(
                    f"swept orphan instance {inst.id} (label={self.label!r})",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARNING: could not destroy orphan {inst.id}: {exc}",
                    file=sys.stderr,
                )

    def _on_signal(self, signum, frame):
        self._destroy_once()
        raise KeyboardInterrupt if signum == signal.SIGINT else SystemExit(128 + signum)

    def __enter__(self) -> _Teardown:
        atexit.register(self._destroy_once)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):  # not main thread / unsupported
                pass
        return self

    def __exit__(self, *exc_info) -> bool:
        self._destroy_once()
        atexit.unregister(self._destroy_once)
        for sig, handler in self._prev.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        return False  # propagate exceptions


def run_spike(
    client: VastClient,
    *,
    image: str,
    candidates: Sequence[str] = DEFAULT_LM_EVAL_CANDIDATES,
    out_path: str = "results/_meta/spike.json",
    max_runtime_s: int = 1800,
    hf_token: str | None = None,
    keep_alive: bool = False,
    offer_query: str = DEFAULT_OFFER_QUERY,
    max_dph: float = DEFAULT_MAX_DPH,
    boot_timeout_s: float = 1800.0,
    remote_script: str | None = None,
    bootstrap_script: str | None = None,
) -> SpikeVerdict:
    """Run the go/no-go spike and return the verdict; always tears the instance down."""
    remote_script = remote_script or str(_repo_path("scripts/spike/spike_remote.py"))
    bootstrap_script = bootstrap_script or str(_repo_path("scripts/spike/bootstrap.sh"))

    offers = client.search_offers(offer_query)
    offer = select_cheapest(offers, max_dph=max_dph)

    env: dict[str, str] = {}
    api_key = os.environ.get("VAST_API_KEY")
    if api_key:
        env["VAST_API_KEY"] = api_key
    if hf_token:
        env["HF_TOKEN"] = hf_token

    # Arm teardown BEFORE create so the create->arm window can't leak the instance.
    with _Teardown(
        client, None, enabled=not keep_alive, label="qquant-spike"
    ) as teardown:
        instance_id = client.create_instance(
            offer.ask_id,
            image=image,
            disk_gb=120,
            onstart_cmd=_deadman_onstart(max_runtime_s),
            env=env,
            label="qquant-spike",
        )
        teardown.instance_id = instance_id
        # The bare CUDA-devel image is large; a slow host can take many minutes to pull
        # it before actual_status reaches "running", so the boot wait is generous.
        wait_until_running(client, instance_id, timeout_s=boot_timeout_s)
        # Bare CUDA images have no /workspace (a vast pytorch-template convention) —
        # create it before copying, else `vastai copy` silently lands nothing there.
        client.ssh_exec(instance_id, "mkdir -p /workspace", timeout_s=120)
        client.copy_to(instance_id, remote_script, REMOTE_SCRIPT)
        client.copy_to(instance_id, bootstrap_script, REMOTE_BOOTSTRAP)

        verdict: SpikeVerdict | None = None
        notes: list[str] = []
        for ref in candidates:
            result = client.ssh_exec(
                instance_id, _remote_command(ref), timeout_s=max_runtime_s
            )
            # The verdict FILE is the source of truth, not the exit code: spike_remote
            # exits 1 on NO-GO yet still writes a full verdict. Always try to copy it
            # back; a missing/unparseable file means bootstrap failed before it ran.
            raw: dict | None = None
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    local = os.path.join(tmp, "verdict.json")
                    client.copy_from(instance_id, REMOTE_VERDICT, local)
                    raw = json.loads(Path(local).read_text())
            except Exception as exc:  # noqa: BLE001
                tail = (result.stderr or result.stdout or "").strip()[-1500:]
                notes.append(
                    f"candidate {ref!r}: no verdict (remote rc={result.returncode}, "
                    f"copy failed: {exc}): {tail}"
                )
                continue
            verdict = verdict_from_payload(raw)
            if verdict.verdict == "GO":
                break

        if verdict is None:
            verdict = _nogo(notes or ["no candidate produced a verdict"])
        elif notes:
            # surface earlier-candidate bootstrap failures alongside the chosen verdict
            verdict = replace(verdict, notes=[*verdict.notes, *notes])
        _write_verdict(out_path, verdict)
        return verdict


def main(argv: list[str] | None = None) -> int:
    """qquant-spike CLI. Exit 0 iff verdict == GO."""
    parser = argparse.ArgumentParser(
        prog="qquant-spike",
        description="Go/no-go GPU spike: validate lm-eval x transformers v5 on a 4090.",
    )
    parser.add_argument(
        "--image", default=QQUANT_IMAGE, help="container image (default: QQUANT_IMAGE)"
    )
    parser.add_argument(
        "--candidates",
        default=",".join(DEFAULT_LM_EVAL_CANDIDATES),
        help="comma-separated lm-eval refs, tried in order",
    )
    parser.add_argument(
        "--out", default="results/_meta/spike.json", help="verdict output path"
    )
    parser.add_argument(
        "--max-runtime-s", type=int, default=1800, help="dead-man's-switch cap"
    )
    parser.add_argument(
        "--keep-alive", action="store_true", help="skip destroy (debug)"
    )
    parser.add_argument(
        "--offer-query", default=DEFAULT_OFFER_QUERY, help="vast.ai offer query"
    )
    parser.add_argument(
        "--max-dph", type=float, default=DEFAULT_MAX_DPH, help="max $/hr offer"
    )
    parser.add_argument(
        "--boot-timeout-s",
        type=float,
        default=1800.0,
        help="seconds to wait for the instance to reach 'running' (slow image pulls)",
    )
    parser.add_argument(
        "--ssh-key",
        default=None,
        help="ssh private key path registered with vast.ai (for copy + ssh-exec)",
    )
    args = parser.parse_args(argv)

    candidates = tuple(c.strip() for c in args.candidates.split(",") if c.strip())
    client = VastClient(ssh_identity=args.ssh_key)
    verdict = run_spike(
        client,
        image=args.image,
        candidates=candidates,
        out_path=args.out,
        max_runtime_s=args.max_runtime_s,
        hf_token=os.environ.get("HF_TOKEN"),
        keep_alive=args.keep_alive,
        offer_query=args.offer_query,
        max_dph=args.max_dph,
        boot_timeout_s=args.boot_timeout_s,
    )
    print(
        f"verdict={verdict.verdict} lm_eval_ref={verdict.lm_eval_ref!r} -> {args.out}"
    )
    for note in verdict.notes:
        print(f"  note: {note}")
    return 0 if verdict.verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
