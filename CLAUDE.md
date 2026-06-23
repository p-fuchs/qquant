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
