"""qquant.eval — GPU-touching quality-eval engine (Spec 05).

Public surface added in Task 9. torch/lm_eval/datasets are imported lazily inside
functions; importing this package stays torch-free. NOT part of the torch-free core —
the umbrella ``qquant`` CLI must never import it.
"""
