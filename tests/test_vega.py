import itertools
import typing
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytest
import vega_datasets
import yaml

from benchmark.datasets import VegaDatasets
from edaplot.data_utils import df_preprocess, spec_remove_data
from edaplot.paths import PATH_RESOURCES
from edaplot.spec_utils import SpecType
from edaplot.vega import (
    AutoToolTip,
    SpecInfo,
    fix_filter_by_date,
    make_text_spec,
    pd_timestamp_to_vl_datetime,
    process_extracted_specs,
    spec_fix_restore_fields,
    spec_set_auto_tooltip,
    validate_and_fix_spec,
    validate_spec,
)
from edaplot.vega_chat.prompts import VEGA_LITE_SCHEMA_URL
from tests.common import get_default_dataset

SPECS_DIR = PATH_RESOURCES / "specs"


def load_resource_dataframe(in_spec: SpecType) -> pd.DataFrame:
    data_field = in_spec["data"]
    if "values" in data_field:  # inline values
        df = pd.DataFrame(data_field["values"])
    else:
        # Custom url format: dataset_name/data_id
        url = data_field["url"]
        ds_name, id_ = url.split("/")
        if ds_name == "data":
            # Allow format: `"data": {"url": "data/cars.yaml"}`
            ds_name = VegaDatasets.name()
            id_ = id_.rsplit(".")[0]
        dataset = get_default_dataset(ds_name)
        df = dataset[id_].data

    # Apply preprocessing like for the model
    # We abuse the "data" field to pass different data params.
    normalize_column_names = data_field.get("normalize_column_names", False)
    parse_dates = data_field.get("parse_dates", True)
    return df_preprocess(df, normalize_column_names=normalize_column_names, parse_dates=parse_dates)


def res_get_resolved_kv(res: dict[str, Any], key: str) -> Any:
    # A key's value can point to another key (e.g. if gt_spec == in_spec)
    key_value = res[key]
    if isinstance(key_value, str) and key_value in res:
        return res[key_value]
    return key_value


def load_yaml_spec_resource(path: Path) -> dict[str, Any]:
    with open(path) as f:
        res = yaml.safe_load(f)
    in_spec = res["in_spec"]
    res["df"] = load_resource_dataframe(in_spec)
    spec_remove_data(in_spec)
    if "gt_spec" in res and (gt_spec := res_get_resolved_kv(res, "gt_spec")) is not None:
        spec_remove_data(gt_spec)
        res["gt_spec"] = gt_spec
    if "spec_history" in res:
        res["spec_history"] = [SpecInfo.from_valid(spec) for spec in res["spec_history"]]
    else:
        res["spec_history"] = []
    return res


fix_success_paths = sorted(SPECS_DIR.glob("fix1_*.yaml"))


@pytest.mark.parametrize("path", fix_success_paths, ids=[p.stem for p in fix_success_paths])
def test_fix_spec_success(path: Path) -> None:
    res = load_yaml_spec_resource(path)
    spec_fix = validate_and_fix_spec(
        res["in_spec"], res["df"], retry_on_empty_plot=True, max_reply_length=100, spec_history=res["spec_history"]
    )
    assert spec_fix.spec is not None
    assert spec_fix.spec == res["gt_spec"]
    assert spec_fix.spec_validity is not None
    assert spec_fix.spec_validity.is_valid_scenegraph
    assert spec_fix.reply is None


fix_fail_paths = sorted(SPECS_DIR.glob("fix0_*.yaml"))


@pytest.mark.parametrize("path", fix_fail_paths, ids=[p.stem for p in fix_fail_paths])
def test_fix_spec_fail(path: Path) -> None:
    res = load_yaml_spec_resource(path)
    spec_fix = validate_and_fix_spec(
        res["in_spec"], res["df"], retry_on_empty_plot=True, max_reply_length=100, spec_history=res["spec_history"]
    )
    assert spec_fix.spec is not None
    assert spec_fix.spec == res["in_spec"]  # unchanged
    assert spec_fix.spec_validity is not None
    assert not spec_fix.spec_validity.is_valid_scenegraph
    assert spec_fix.reply is not None


fix_transform_paths = sorted(SPECS_DIR.glob("fix_transform_*.yaml"))


@pytest.mark.parametrize("path", fix_transform_paths, ids=[p.stem for p in fix_transform_paths])
def test_fix_transforms(path: Path) -> None:
    res = load_yaml_spec_resource(path)
    spec_fix = validate_and_fix_spec(
        res["in_spec"], res["df"], retry_on_empty_plot=True, max_reply_length=100, spec_history=res["spec_history"]
    )
    assert spec_fix.spec is not None
    assert spec_fix.spec == res["in_spec"]  # unchanged
    assert spec_fix.spec_validity is not None
    assert spec_fix.reply is not None  # crucial


validate_paths = sorted(SPECS_DIR.glob("val_*.yaml"))


@pytest.mark.parametrize("path", validate_paths, ids=[p.stem for p in validate_paths])
def test_validate_spec(path: Path) -> None:
    res = load_yaml_spec_resource(path)
    got = validate_spec(res["in_spec"], res["df"])
    for k, expected_v in res["gt_validity"].items():
        assert getattr(got, k) == expected_v, f"Mismatch on {k=}"


fix_restore_fields_paths = sorted(SPECS_DIR.glob("fix_restore_fields_*.yaml"))


@pytest.mark.parametrize("path", fix_restore_fields_paths, ids=[p.stem for p in fix_restore_fields_paths])
def test_fix_restore_fields(path: Path) -> None:
    res = load_yaml_spec_resource(path)
    spec_validity = validate_spec(res["in_spec"], res["df"])
    prev_spec = res["spec_history"][-1]
    spec_fix = spec_fix_restore_fields(res["in_spec"], res["df"], spec_validity, prev_spec)
    assert spec_fix.spec == res_get_resolved_kv(res, "fix_restore_fields_spec")


@pytest.mark.parametrize(
    ["specs", "expected"],
    [
        ([{"a": 1, "b": None, "data": {"url": "data.csv"}}], {"$schema": VEGA_LITE_SCHEMA_URL, "a": 1, "b": None}),
        # Test for when the LLM generates multiple comma-separated JSONs instead of one (TODO)
        (
            [{"a": 1, "b": None, "data": {"url": "data.csv"}}, {"a": 2, "b": None, "data": {"url": "data.csv"}}],
            {"$schema": VEGA_LITE_SCHEMA_URL, "hconcat": [{"a": 1}, {"a": 2}]},
        ),
    ],
)
def test_process_extracted_specs(specs: list[SpecType], expected: SpecType) -> None:
    if len(specs) > 1:
        with pytest.raises(ValueError):
            process_extracted_specs(specs)
    else:
        got = process_extracted_specs(specs)
        assert got == expected


@pytest.mark.parametrize(
    "inp, expected",
    [
        (
            {"filter": {"field": "Year", "gte": "2010-06-01"}},
            {"filter": {"field": "Year", "gte": pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01"))}},
        ),
        ({"filter": {"field": "Year", "gte": 2010}}, {"filter": {"field": "Year", "gte": {"year": 2010}}}),
        (
            {"filter": {"field": "Year", "gte": "2010-06-01"}},
            {"filter": {"field": "Year", "gte": pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01"))}},
        ),
        (
            {"filter": {"field": "Year", "timeUnit": "month", "gte": "2010-06-01"}},
            {"filter": {"field": "Year", "timeUnit": "month", "gte": 6}},
        ),
        (
            {"filter": {"field": "Year", "timeUnit": "year", "gte": "2010-06-01"}},
            {"filter": {"field": "Year", "timeUnit": "year", "gte": 2010}},
        ),
        ({"filter": {"field": "Year", "timeUnit": "year", "gte": 2010}}, "same"),
        ({"filter": {"field": "Year", "timeUnit": "month", "gte": 6}}, "same"),
        (
            {"filter": {"and": [{"field": "Year", "gte": "2010-06-01"}, {"field": "Year", "lte": "2010-06-03"}]}},
            {
                "filter": {
                    "and": [
                        {"field": "Year", "gte": pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01"))},
                        {"field": "Year", "lte": pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-03"))},
                    ]
                }
            },
        ),
        ({"filter": "datum.Year > 2012"}, "same"),
        (
            {"filter": {"not": {"field": "Year", "equal": "2010-06-01"}}},
            {"filter": {"not": {"field": "Year", "equal": pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01"))}}},
        ),
        ({"filter": {"not": {"field": "Year", "equals": "2010-06-01"}}}, "reply"),  # equals is not valid
        ({"filter": {"not": {"field": "Year", "equal": {"year": 2010}}}}, "same"),  # keep valid datetimes
        ({"filter": {"field": "Some field", "equal": "cool"}}, "same"),  # ignore non-dates
        ({"filter": {"field": "date", "equal": "2010"}}, "same"),  # ignore non-dates
        ({"filter": {"field": "Year", "gte": "Not a date"}}, "reply"),  # parse error
        ({"filter": {"field": "Year", "gte": "192.168.1.1"}}, "reply"),  # parse error
        (
            {"filter": {"field": "Year", "gte": "2020.04"}},
            {"filter": {"field": "Year", "gte": pd_timestamp_to_vl_datetime(pd.to_datetime("2020.04"))}},
        ),
        (
            {"filter": {"field": "Year", "gte": "2010-06-01T00:00:00Z"}},
            {"filter": {"field": "Year", "gte": pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01T00:00:00Z"))}},
        ),
        ({"filter": {"field": "Year", "gte": "2024"}}, {"filter": {"field": "Year", "gte": {"year": "2024"}}}),
        ({"filter": {"field": "Year", "gte": "6"}}, "reply"),
        ({"filter": {"field": "Year", "gte": 6}}, "reply"),
        ({"filter": {"field": "Year", "gte": 123}}, "reply"),
        ({"filter": {"field": "Year", "gte": None}}, "reply"),
        ({"filter": {"field": "Year", "gte": {"year": 2018}}}, "same"),
        (
            {"filter": {"field": "Year", "range": ["2010-06-01T00:00:00Z", "2018-06-01T00:00:00Z"]}},
            {
                "filter": {
                    "field": "Year",
                    "range": [
                        pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01T00:00:00Z")),
                        pd_timestamp_to_vl_datetime(pd.to_datetime("2018-06-01T00:00:00Z")),
                    ],
                }
            },
        ),
        (
            {"filter": {"field": "Year", "range": [2010, 2018]}},
            {"filter": {"field": "Year", "range": [{"year": 2010}, {"year": 2018}]}},
        ),
        ({"filter": {"field": "Year", "range": [{"year": 2010}, {"year": 2018}]}}, "same"),
        ({"filter": {"field": "Year", "range": [5, 10]}}, "reply"),
        ({"filter": {"field": "Year", "range": 5}}, "reply"),
        (
            {
                "filter": {
                    "or": [
                        {"field": "Year", "range": ["2010-06-01", "2015-06-01"]},
                        {"field": "Year", "equal": "2020-06-03"},
                    ]
                }
            },
            {
                "filter": {
                    "or": [
                        {
                            "field": "Year",
                            "range": [
                                pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01")),
                                pd_timestamp_to_vl_datetime(pd.to_datetime("2015-06-01")),
                            ],
                        },
                        {"field": "Year", "equal": pd_timestamp_to_vl_datetime(pd.to_datetime("2020-06-03"))},
                    ]
                }
            },
        ),
        (
            {"filter": {"field": "Year", "oneOf": ["2010-06-01T00:00:00Z", "2018-06-01T00:00:00Z"]}},
            {
                "filter": {
                    "field": "Year",
                    "oneOf": [
                        pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01T00:00:00Z")),
                        pd_timestamp_to_vl_datetime(pd.to_datetime("2018-06-01T00:00:00Z")),
                    ],
                }
            },
        ),
        (
            {"filter": {"field": "Year", "oneOf": [2018]}},
            {"filter": {"field": "Year", "oneOf": [{"year": 2018}]}},
        ),
        ({"filter": {"field": "Year", "oneOf": [{"year": 2010}, {"year": 2018}]}}, "same"),
        ({"filter": {"field": "Year", "oneOf": [5, 10]}}, "reply"),
        ({"filter": {"field": "Year", "oneOf": 5}}, "reply"),
        (
            {
                "filter": {
                    "or": [
                        {"field": "Year", "range": ["2010-06-01", "2015-06-01"]},
                        {"field": "Year", "oneOf": ["2020-06-03"]},
                    ]
                }
            },
            {
                "filter": {
                    "or": [
                        {
                            "field": "Year",
                            "range": [
                                pd_timestamp_to_vl_datetime(pd.to_datetime("2010-06-01")),
                                pd_timestamp_to_vl_datetime(pd.to_datetime("2015-06-01")),
                            ],
                        },
                        {"field": "Year", "oneOf": [pd_timestamp_to_vl_datetime(pd.to_datetime("2020-06-03"))]},
                    ]
                }
            },
        ),
    ],
)
def test_fix_filter_by_date(inp: dict, expected: dict | Literal["same", "reply"]) -> None:
    df = vega_datasets.data("cars")  # df["Year"] is a datetime64[ns] column
    is_fixed, new_filter, reply = fix_filter_by_date({}, df, inp)
    if isinstance(expected, dict):
        assert is_fixed
        assert new_filter == expected
        assert reply is None
    else:
        assert not is_fixed
        assert new_filter == inp
        if expected == "reply":
            assert reply is not None


@pytest.mark.parametrize(
    "spec",
    [
        {
            "mark": "point",
            "encoding": {"x": {"field": "Horsepower", "type": "quantitative"}, "tooltip": {"field": "Horsepower"}},
        },
        {
            "mark": {"type": "point"},
            "encoding": {"x": {"field": "Horsepower", "type": "quantitative"}, "tooltip": {"field": "Horsepower"}},
        },
        {
            "mark": {"type": "point", "tooltip": True},
            "encoding": {
                "x": {"field": "Horsepower", "type": "quantitative"},
            },
        },
    ],
)
def test_spec_set_auto_tooltip(spec: SpecType) -> None:
    for tooltip_type, force in itertools.product(typing.get_args(AutoToolTip), [True, False]):
        got = spec_set_auto_tooltip(spec, tooltip_type=tooltip_type, replace_tooltip=force)
        match tooltip_type:
            case "data" | "encoding" as tt_type:
                assert got["mark"]["tooltip"] == {"content": tt_type}
            case "none":
                assert got["mark"]["tooltip"] is None
            case _:
                assert spec == got
        if force:
            assert "tooltip" not in got["encoding"]
        else:
            assert spec["encoding"] == got["encoding"]


def test_make_text_spec() -> None:
    spec = make_text_spec("Hello world")
    spec_validity = validate_spec(spec, pd.DataFrame())
    assert spec_validity.is_valid_scenegraph
    assert spec_validity.is_valid_schema
    assert spec_validity.is_empty_scenegraph
