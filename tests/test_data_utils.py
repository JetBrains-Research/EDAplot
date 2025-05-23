from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from benchmark.datasets import Dataset
from edaplot.data_utils import df_convert_types, df_is_datetime_column, df_parse_dates, df_preprocess
from tests.test_datasets import default_dataset


@pytest.mark.parametrize(
    "data, expected_col",
    [
        ({"date": ["2021-01-01", "2021-02-01", "2021-03-01"]}, "date"),
        ({"date": ["2021-01-01", None, "2021-03-01"]}, "date"),
        ({"date": ["2021/01/01", "2021/01/02", "2021/01/03"]}, "date"),
        ({"date": ["March 3, 2021", "March 2, 2021", "March 1, 2021"]}, "date"),
        ({"date": [1349720105, 1349806505, 1349892905, 1349979305, 1350065705]}, None),
        ({"year": [1, 2, 3]}, None),  # pd.to_datetime(1) will not be a year...
        ({"year": [2021, 2022]}, None),
        ({"year": [Decimal(2024)]}, None),
        ({"year": [Decimal(2024.0)]}, None),
        ({"date": ["2021/01/01", "02-01-2021", "March 1, 2021"]}, None),
        ({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}, None),
        ({"date": ["2021-01-01", "not a date", "2021-03-01"]}, None),
        (
            {
                "Year": ["2021", "2022", "2023"],
                "EventTime": ["12:00", "13:00", "14:00"],
                "Timestamp": ["2021-01-01 12:00", "2021-02-01 13:00", "2021-03-01 14:00"],
            },
            "Timestamp",
        ),
    ],
)
def test_parse_dates(data: dict, expected_col: str) -> None:
    df = pd.DataFrame(data)
    got = df_parse_dates(df)
    if expected_col:
        assert df_is_datetime_column(got, expected_col)
    else:
        for col in data.keys():
            assert not df_is_datetime_column(got, col)


def test_preprocess_dataframe(default_dataset: Dataset) -> None:
    # Check that data preprocessing doesn't crash or something
    for item in default_dataset:
        new_df = df_preprocess(item.data, normalize_column_names=True, parse_dates=True, convert_types=True)
        assert new_df.shape == item.data.shape


@pytest.mark.parametrize(
    "data, expected_type, expected_value",
    [
        ({"value": [Decimal("10")]}, "numeric", [10.0]),
        ({"value": [Decimal("10.0")]}, "numeric", [10.0]),
        ({"value": [Decimal("10.5")]}, "numeric", [10.5]),
        ({"value": [Decimal("10"), Decimal("20")]}, "numeric", [10, 20]),
        ({"value": [Decimal("10.5"), Decimal("20.5")]}, "numeric", [10.5, 20.5]),
        ({"value": [Decimal("10"), Decimal("20.5")]}, "numeric", [10.0, 20.5]),
        ({"value": ["10", "20"]}, object, ["10", "20"]),  # Not Decimal objects
        ({"value": [10, 20]}, int, [10, 20]),  # Already integers
        ({"value": [10.5, 20.5]}, float, [10.5, 20.5]),  # Already floats
        ({"value": [Decimal("10"), None, Decimal("20")]}, "numeric", [10, None, 20]),  # With None values
    ],
)
def test_convert_types(data: dict, expected_type: Any, expected_value: list) -> None:
    df = pd.DataFrame(data)
    got = df_convert_types(df)
    if expected_type == "numeric":
        assert pd.api.types.is_numeric_dtype(got["value"].dtype)
    else:
        assert got["value"].dtype == expected_type or pd.api.types.is_dtype_equal(got["value"].dtype, expected_type)
    pd.testing.assert_series_equal(
        got["value"].fillna(pd.NA), pd.Series(expected_value, name="value").fillna(pd.NA), check_dtype=False
    )
