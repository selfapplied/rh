
import pytest
from tools import run_refraction_scan


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(zero_t=14.134725, depth=0, facets=8, window=0.5, step=0.1, d=0.05, gamma=1.0),
        dict(zero_t=14.134725, depth=3, facets=2, window=0.5, step=0.1, d=0.05, gamma=1.0),
        dict(zero_t=14.134725, depth=3, facets=8, window=0.0, step=0.1, d=0.05, gamma=1.0),
        dict(zero_t=14.134725, depth=3, facets=8, window=0.5, step=0.0, d=0.05, gamma=1.0),
        dict(zero_t=14.134725, depth=3, facets=8, window=0.2, step=1.0, d=0.05, gamma=1.0),  # step > 2*window
        dict(zero_t=float("inf"), depth=3, facets=8, window=0.5, step=0.1, d=0.05, gamma=1.0),
        dict(zero_t=14.134725, depth=3, facets=8, window=0.5, step=0.1, d=float("nan"), gamma=1.0),
        dict(zero_t=14.134725, depth=3, facets=8, window=0.5, step=0.1, d=0.05, gamma=0.0),
    ],
)
def test_run_refraction_scan_param_validation(kwargs):
    with pytest.raises(ValueError):
        run_refraction_scan(**kwargs)


def test_run_refraction_scan_sanity():
    res = run_refraction_scan(
        zero_t=14.134725,
        depth=3,
        facets=8,
        window=0.2,
        step=0.2,
        d=0.05,
        gamma=1.0,
    )
    assert isinstance(res, dict)
    assert "spectrum" in res and isinstance(res["spectrum"], list)
    assert len(res["spectrum"]) == 8
    assert all(isinstance(float(x), float) for x in res["spectrum"])  # castable to float
    assert 0 <= int(res["locked_orientation"]) < 8
    assert res.get("N", 0) > 0
    assert len(res.get("mask", [])) == res["N"]
    assert len(res.get("template", [])) == res["N"]