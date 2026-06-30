"""qquant-ext CLI: the gated registry yields empty plans without importing torch."""

from __future__ import annotations

from qquant.ext.cli import main


def test_plan_on_gated_registry_is_empty(capsys):
    """All shipped ext variants are enabled:false -> plan prints nothing, rc 0."""
    rc = main(["--results", "results-ext", "plan"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_eval_on_gated_registry_is_noop(capsys):
    rc = main(["--results", "results-ext", "eval", "--verbose"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "written=0" in out


def test_unknown_profile_variant_rejected(capsys):
    rc = main(["profile", "--variant", "not-real"])
    assert rc == 2
