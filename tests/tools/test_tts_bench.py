"""Text normalisation, error rates and benchmark summaries."""

import math

from tools.tts_bench.bench import summarize
from tools.tts_bench.metrics import bootstrap_ci, cer, normalize, wer


def test_normalize_drops_punctuation_danda_and_zero_width():
    assert normalize("नमस्ते,  साथी।‍ ") == "नमस्ते साथी"


def test_error_rates():
    assert cer("नमस्ते साथी", "नमस्ते साथी।") == 0.0
    assert wer("एक दुई तीन", "एक तीन") == 1 / 3
    # word-boundary differences do not count as character errors
    assert cer("घर भित्र", "घरभित्र") == 0.0
    assert wer("घर भित्र", "घरभित्र") == 1.0


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
    assert lo <= mean == 3 <= hi
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_summarize_pairs_models_by_item():
    def row(model, item, c, seen=False):
        return {"model": model, "item": item, "seen": seen, "voice": "female",
                "cer": c, "wer": c, "sim": 0.5, "utmos": 3.0}
    rows = [row("base", "a", 0.2), row("base", "b", 0.4), row("run1", "a", 0.1), row("run1", "b", 0.5)]
    s = summarize(rows)
    paired = s["paired_vs_base"]["run1"]
    assert paired["n"] == 2
    assert abs(paired["cer"]["diff"][0] - 0.0) < 1e-9
    assert paired["cer"]["win_rate"] == 0.5
    assert s["groups"]["unseen voices"]["run1"]["n"] == 2
