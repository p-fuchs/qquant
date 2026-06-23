# Project guidance for AI agents (qquant)

The Qwen2.5-7B quantization study. Specs are the source of truth: read
`docs/specs/contracts.md` (SSOT for ids/paths/schema/predicates) and
`docs/specs/decision-log.md` (pins, platform, locked decisions) first. GPU work runs on a
rented **vast.ai RTX 4090**; the orchestration layer (`qquant.orchestrate`) is torch-free
so it runs on the dev laptop.

## DIRECTIVE: record vast.ai learnings here

vast.ai's CLI and runtime have many non-obvious behaviors that have already cost real
debugging and real GPU spend. **Whenever you discover something non-obvious about vast.ai
— CLI flags, instance runtime, auth, networking, the bare-image environment — append it to
the "vast.ai operational notes" section below** so the next agent doesn't rediscover it.
Keep entries terse and concrete.

## Running the spike / orchestration

- Secrets live in `.env` (git-ignored). `VAST_API_KEY` is required. Load it for one
  command: `set -a; . ./.env; set +a; uv run qquant-spike ...`. The key must be in the
  **environment** (not only vastai's config file) so the on-instance dead-man's switch is
  injected with it.
- The SSH private key registered with vast.ai lives at `ssh_keys/vastai` (git-ignored).
  Pass it: `uv run qquant-spike --ssh-key ssh_keys/vastai`.
- `qquant-spike` exits 0 iff the verdict is GO; the verdict JSON goes to `--out`
  (default `results/_meta/spike.json`).
- **Never commit** `.env` or anything under `ssh_keys/` (both git-ignored).

## vast.ai operational notes (verified against vastai 1.1.1)

- **Auth**: CLI reads `VAST_API_KEY` from env or `~/.config/vastai/vast_api_key`
  (`vastai set api-key`). `vastai show user --raw` exposes `balance`/`credit`; the
  available credit is itself a hard worst-case spend cap.
- **`destroy` needs `-y`**: `vastai destroy instance <id> -y`. Without `-y` it prompts, and
  a non-interactive call hangs → leaked billing instance.
- **`show instances` is deprecated** (use `show instances-v1` for the paginated form). The
  DEPRECATED warning prints to **stderr**, so `show instances --raw` stdout is still clean
  (`[]` when none). `instances-v1 --raw` returns an **object** `{"instances": [...], ...}`,
  not a bare list.
- **Offer query**: encode spaces in string values as `_` (e.g. `gpu_name=RTX_4090`). A
  default query (`external=false rentable=true verified=true`) is ANDed unless you pass
  `-n`. `vastai create instance` returns the new id as `new_contract` in `--raw` JSON.
- **The bare `nvidia/cuda:*-devel` image has no python/pip/torch** — install everything via
  `uv` at bootstrap (torch from the cu126 index, pinned to the decision-log version). The
  dead-man's switch therefore cannot use pip; it self-destroys via the REST API with
  `curl -X DELETE /api/v0/instances/$CONTAINER_ID/ -H "Authorization: Bearer $VAST_API_KEY"`.
- **The bare image has no `/workspace`** — that directory is a vast *pytorch-template*
  convention. `mkdir -p /workspace` before using it (in the onstart and before `vastai
  copy`); copying to a non-existent remote dir silently lands NOTHING yet returns rc 0
  (the failure surfaces later as `bash: /workspace/...: No such file` → exit 127).
- **Boot is slow**: pulling the large CUDA-devel image can take >15 min on some hosts;
  `actual_status` stays `None`/pre-running meanwhile. Use a generous boot timeout
  (`qquant-spike --boot-timeout-s`, default 1800).
- **vast prints a long MOTD to stderr** on ssh login ("Welcome to vast.ai... Have fun!");
  capture enough of the tail when surfacing remote errors that the real message survives.
- **`vastai copy` is unreliable for local↔instance**: it returns rc 0 but transfers
  nothing to a bare-image instance. Use `scp -i <key>` over the `ssh-url` host/port (the
  same path `ssh` uses) instead — it actually transfers and returns real error codes.
- **Bootstrap gotchas (stack)**: `gptqmodel 7.1.0` + its deps `tokenicer 0.0.13` /
  `logbar 0.4.3` are **sdist-only** on PyPI and use the PEP 639 string license
  (`license = "Apache-2.0"`), so they **require `setuptools>=77.0.1,<83`** to build
  (setuptools<77 rejects the string license with "project.license must be valid exactly by
  one definition"). Since the install is `--no-build-isolation`, pin the venv's setuptools
  into that range BEFORE installing. Make the bootstrap idempotent across retries/candidate
  refs (`uv venv` errors if the venv already exists; guard with
  `[ -x "$VENV/bin/python" ] || uv venv ...`). Verified the boundary locally for $0 by
  test-building the pure-python sdists across setuptools versions (use `--no-cache` — uv
  caches built wheels by name+version, not by setuptools version).
- **`gptqmodel 7.1.0` needs `torchvision`** at model-load time but does NOT declare it (not
  in the lock) — install `torchvision` (cu126 index, matches torch 2.12.1 → 0.27.1+cu126)
  or gptq/awq loads die with `ModuleNotFoundError: No module named 'torchvision'`.
- **`gptqmodel` JIT-compiles its sm_89 Marlin torch.ops extension at first load and
  hard-requires `ninja`** ("Ninja is required to load C++ extensions"); install `ninja`
  (+ a C/C++ toolchain via `build-essential`, + nvcc on PATH) or gptq/awq loads fail.
- **gsm8k dataset id**: lm-eval's gsm8k task uses the bare `gsm8k` repo id, which was
  renamed to `openai/gsm8k`; `datasets>=4` (and 3.x) reject the bare id with `HfUriError`
  (verified locally — only `openai/gsm8k` resolves). The real GSM8K eval must use
  `openai/gsm8k` (patch the task yaml / rewrite at the datasets layer).
- **ssh sessions do NOT inherit the image's PATH** — `export PATH=/usr/local/cuda/bin:$PATH`
  for nvcc and gptqmodel's sm_89 JIT build. `--env` vars are also NOT visible in the ssh
  session unless the onstart persists them (`env | grep _ >> /etc/environment`); the onstart
  script itself DOES see the injected `--env` vars.
- **SSH keys**: register the public key with `vastai create ssh-key "$(cat key.pub)" -y`.
  Both `vastai copy` and `ssh` need the **private** key: `vastai copy -i <key>` and
  `ssh -i <key>`. Keys outside `~/.ssh` need explicit `-i` (wired through `qquant-spike
  --ssh-key`). Do NOT copy to `/root` or `/` (breaks the instance's ssh perms).
- **`git ls-remote` rejects a `git+https://` URL** (that scheme is pip-only): strip `git+`
  for git, keep it for pip. Under `set -euo pipefail`, a failing `$(...)` command
  substitution aborts the script *before* any fallback line — guard with `|| true`.
- **`$CONTAINER_ID`** is *assumed* to be the numeric instance id used by the self-destruct
  REST call — **verify on the box** (it may be the docker hash); the deadman logs to
  `/workspace/deadman.log`.

## Commands

- `uv sync` · `uv run pytest` · `uv run ruff check . && uv run ruff format --check .`
- Torch-free guard: `qquant.*` and `qquant.orchestrate.*` must never import torch (enforced
  by `tests/test_import_torch_free.py` + CI). GPU code lives only in `scripts/spike/`.
