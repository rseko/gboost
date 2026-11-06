"""Tests for the py-gboost (gboostrs) Python bindings."""

import numpy as np
import pytest
from gboostrs import GradientBooster


def test_options():
    gb = GradientBooster()
    assert len(gb.get_params()) == 21

    copy = GradientBooster(**gb.get_params())
    assert copy.get_params() == gb.get_params()

    with pytest.raises(ValueError):
        GradientBooster(objective="")
    with pytest.raises(ValueError):
        GradientBooster(grow_policy="")


def test_regression():
    rng, n, d = np.random.default_rng(1), 100, 5
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    gb = GradientBooster(objective="squaredloss", iterations=10)
    gb.fit(x, y)
    pred = gb.predict(x)

    assert np.allclose(pred[:3], [-0.94069624, 1.352104, -0.857982])
    assert np.allclose(gb.loss(), 0.042405)


def test_classification():
    rng, n, d = np.random.default_rng(1), 100, 5
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.float32)

    gb = GradientBooster(objective="logloss", iterations=10)
    gb.fit(x, y)
    pred = gb.predict(x)

    assert np.allclose(pred[:3], [2.947796, 1.9830728, 2.0805707])
    assert np.allclose(gb.loss(), 0.18994)


def test_fit_next():
    rng, n, d = np.random.default_rng(1), 200, 5
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    gb = GradientBooster(objective="squaredloss", iterations=0)
    gb.fit(x, y)

    prev = gb.loss()
    for _ in range(10):
        assert gb.fit_next() is True
        cur = gb.loss()
        assert cur < prev
        prev = cur


def test_fit_failures():
    rng, n, d = np.random.default_rng(1), 100, 5
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    # A target containing NaN is rejected.
    y_nan = y.copy()
    y_nan[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        GradientBooster().fit(x, y_nan)

    # A target whose length differs from the number of rows is rejected.
    with pytest.raises(ValueError, match="size"):
        GradientBooster().fit(x, y[:10])

    # A persist_path whose parent does not exist maps to an IOError.
    with pytest.raises(IOError):
        GradientBooster(persist_path="\0").fit(x, y)


def test_predict_failures():
    rng, n, d = np.random.default_rng(1), 100, 5
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    gb = GradientBooster()
    gb.fit(x, y)

    # Predicting with a different number of columns than training is rejected.
    with pytest.raises(ValueError, match="columns"):
        gb.predict(x[:, : d - 1])


def test_untrained_guards():
    x = np.zeros((100, 5), dtype=np.float32)
    gb = GradientBooster()

    # Before fit: loss is None, predict/fit_next raise.
    assert gb.loss() is None
    with pytest.raises(ValueError, match="not trained"):
        gb.predict(x)
    with pytest.raises(ValueError, match="not trained"):
        gb.fit_next()
