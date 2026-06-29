"""Runtime dataset-id overrides for lm-eval. Torch-free; imports `datasets` lazily.

lm-eval's gsm8k task references the bare `gsm8k` repo id, which datasets>=4 rejects
(HfUriError); the working id is `openai/gsm8k`. apply_dataset_overrides()
monkeypatches the datasets loaders so the rewrite happens transparently at eval time
(mirrors the Spec-02 spike).
"""

from __future__ import annotations

DATASET_OVERRIDES = {"gsm8k": "openai/gsm8k"}

_PATCH_FLAG = "_qquant_dataset_overrides_applied"


def _wrap(orig):
    def wrapped(path, *args, **kwargs):
        if path in DATASET_OVERRIDES:
            path = DATASET_OVERRIDES[path]
            kwargs.pop("revision", None)  # the pinned revision belonged to the old id
        return orig(path, *args, **kwargs)

    return wrapped


def apply_dataset_overrides(datasets_module=None):
    """Idempotently patch load_dataset/load_dataset_builder to rewrite bare ids."""
    if datasets_module is None:
        import datasets as datasets_module
    if getattr(datasets_module, _PATCH_FLAG, False):
        return datasets_module
    for fn_name in ("load_dataset", "load_dataset_builder"):
        orig = getattr(datasets_module, fn_name)
        setattr(datasets_module, fn_name, _wrap(orig))
    setattr(datasets_module, _PATCH_FLAG, True)
    return datasets_module
