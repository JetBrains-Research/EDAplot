import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.metrics import compute_f1_weighted, compute_spec_metrics, get_my_encoding_fields
from edaplot.paths import PATH_RESOURCES
from edaplot.vega import SpecType


@dataclass
class SpecResource:
    ref_spec: SpecType
    hyp_spec: SpecType
    ref_metrics: dict[str, float]
    hyp_path: Path


def load_spec_resources() -> list[SpecResource]:
    specs_dir = PATH_RESOURCES / "metrics"
    resources = []
    for hyp_path in specs_dir.glob("*.hyp"):
        ref_path = hyp_path.with_suffix(".ref")
        metrics_path = hyp_path.with_suffix(".metrics")
        hyp_spec = json.loads(hyp_path.read_text())
        ref_spec = json.loads(ref_path.read_text())
        ref_metrics = json.loads(metrics_path.read_text())
        res = SpecResource(ref_spec=ref_spec, hyp_spec=hyp_spec, ref_metrics=ref_metrics, hyp_path=hyp_path)
        resources.append(res)
    return resources


spec_resources = load_spec_resources()


@pytest.fixture(params=spec_resources, ids=[r.hyp_path.name for r in spec_resources])
def spec_res(request: Any) -> SpecResource:
    return request.param


@pytest.mark.parametrize(
    ["inp", "include_titles", "out"],
    [
        ({}, True, []),
        (
            {"encoding": {"x": {"field": "date", "title": "Date"}, "y": {"axis": {"title": "Price"}}}},
            True,
            [("x", "field", "date"), ("x", "title", "Date"), ("y", "title", "Price")],
        ),
        (
            {"encoding": {"x": {"field": "date", "title": "Date"}, "y": {"axis": {"title": "Price"}}}},
            False,
            [("x", "field", "date")],
        ),
        (
            {
                "some": [
                    0,
                    {
                        "deep": {
                            "encoding": {"x": {"field": "date", "title": "Date"}, "y": {"axis": {"title": "Price"}}}
                        }
                    },
                ]
            },
            True,
            [("x", "field", "date"), ("x", "title", "Date"), ("y", "title", "Price")],
        ),
    ],
)
def test_get_my_encoding_fields(inp: SpecType, include_titles: bool, out: list[tuple]) -> None:
    got = get_my_encoding_fields(inp, include_titles=include_titles)
    assert got == out


@pytest.mark.parametrize(
    ["hyp", "ref", "f1"],
    [
        ([], [], 1.0),
        ([], [("a", 2.0)], 0.0),
        ([("a", 2.0)], [], 0.0),
        ([("a", 2.0)], [("a", 2.0)], 1.0),
        ([("a", 0.5)], [("a", 2.0)], 0.4),
        ([("a", 0.5), ("a", 0.5)], [("a", 1.0), ("b", 2.0)], 0.5),
    ],
)
def test_compute_f1_weighted(hyp: list, ref: list, f1: float) -> None:
    assert compute_f1_weighted(ref, hyp, beta=1.0).f1 == pytest.approx(f1)


def test_spec_metrics(spec_res: SpecResource) -> None:
    ref_metrics = spec_res.ref_metrics
    hyp_metrics = compute_spec_metrics(spec_res.ref_spec, spec_res.hyp_spec)
    assert set(ref_metrics.keys()).issubset(hyp_metrics.keys()), f"'{spec_res.hyp_path}' contains unknown metrics!"
    for metric_name, hyp_value in hyp_metrics.items():
        if metric_name not in ref_metrics:
            warnings.warn(f"Metric '{metric_name}' not found in '{spec_res.hyp_path}'")
            continue
        assert not np.isnan(hyp_value)
        assert hyp_value == pytest.approx(ref_metrics[metric_name]), metric_name
