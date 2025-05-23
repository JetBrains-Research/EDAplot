import json
import warnings
from typing import Any, Literal, assert_never

import numpy as np
import pandas as pd

from edaplot.data_utils import df_is_datetime_column


def lida_df_description(df_to_descr: pd.DataFrame, n_samples: int = 3) -> str:
    # Copied from https://github.com/JetBrains-Research/PandasPlotBench
    def check_type(dtype: Any, value: Any) -> Any:
        """Cast value to right type to ensure it is JSON serializable"""
        if np.isnan(value):
            return 0
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    properties_list = []
    for column in df_to_descr.columns:
        dtype = df_to_descr[column].dtype
        properties: dict[str, Any] = {}
        if dtype in [int, float, complex]:
            properties["dtype"] = "number"
            properties["std"] = check_type(dtype, df_to_descr[column].std())
            properties["min"] = check_type(dtype, df_to_descr[column].min())
            properties["max"] = check_type(dtype, df_to_descr[column].max())
        elif dtype == bool:
            properties["dtype"] = "boolean"
        elif dtype == object:
            # Check if the string column can be cast to a valid datetime
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(df_to_descr[column], errors="raise")
                    properties["dtype"] = "date"
            except ValueError:
                # Check if the string column has a limited number of values
                if df_to_descr[column].nunique() / len(df_to_descr[column]) < 0.5:
                    properties["dtype"] = "category"
                else:
                    properties["dtype"] = "string"
        elif isinstance(dtype, pd.CategoricalDtype):
            properties["dtype"] = "category"
        elif pd.api.types.is_datetime64_any_dtype(df_to_descr[column]):
            properties["dtype"] = "date"
        else:
            properties["dtype"] = str(dtype)

        # add min max if dtype is date
        if properties["dtype"] == "date":
            try:
                properties["min"] = df_to_descr[column].min().isoformat()
                properties["max"] = df_to_descr[column].max().isoformat()
            except TypeError:
                cast_date_col = pd.to_datetime(df_to_descr[column], errors="coerce")
                properties["min"] = cast_date_col.min().isoformat()
                properties["max"] = cast_date_col.max().isoformat()
        # Add additional properties to the output dictionary
        nunique = df_to_descr[column].nunique()
        if "samples" not in properties:
            non_null_values = df_to_descr[column][df_to_descr[column].notnull()].unique()
            n_samples = min(n_samples, len(non_null_values))
            samples = pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
            properties["samples"] = samples
        properties["num_unique_values"] = nunique
        properties_list.append({"column": column, "properties": properties})

    return json.dumps(properties_list)


def head_df_description(df_to_descr: pd.DataFrame) -> str:
    # Copied from https://github.com/JetBrains-Research/PandasPlotBench
    # The problem with the "head" description strategy is that dates are formatted as epoch timestamps and floats can
    # use lots of tokens with unnecessary decimal places.
    descr_string = f"DataFrame with {len(df_to_descr.columns)} columns and {len(df_to_descr)} rows.\n"
    descr_string += "Column names and types: " + ", ".join(
        [f"{column}: {df_to_descr[column].dtype}" for column in df_to_descr.columns]
    )
    descr_string += "\nHead:\n"
    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        10000,
    ):
        head = str(df_to_descr.head().to_json())
    descr_string += head
    return descr_string


def describe_df_description(df_to_descr: pd.DataFrame) -> str:
    # Copied from https://github.com/JetBrains-Research/PandasPlotBench
    descr_string = f"DataFrame with {len(df_to_descr.columns)} columns " f"and {len(df_to_descr)} rows\n"
    descr_string += "Column names and types: " + ", ".join(
        [f"{column}: {df_to_descr[column].dtype}" for column in df_to_descr.columns]
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        10000,
    ):
        descr = str(df_to_descr.describe().to_json())
    descr_string += "\nColumns stats:\n" + descr
    return descr_string


def df_column_info(df: pd.DataFrame, column: str, n_samples: int = 3) -> dict[str, Any]:
    # Ideas from:
    # https://github.com/microsoft/lida/blob/main/lida/components/summarizer.py
    # https://github.com/cmudig/draco2/blob/main/draco/schema.py
    col = df[column]
    info: dict[str, Any] = {
        "field": column,
        "dtype": str(col.dtype),  # TODO infer VL field type (temporal, quantitative,...) directly?
        "head": col.head(n_samples).to_list(),
    }

    if df_is_datetime_column(df, column):
        try:
            info["min"] = col.min()
            info["max"] = col.max()
        except TypeError:
            cast_date_col = pd.to_datetime(col, errors="coerce")
            info["min"] = cast_date_col.min()
            info["max"] = cast_date_col.max()
    elif pd.api.types.is_numeric_dtype(col.dtype):
        info["min"] = col.min()
        info["max"] = col.max()

    if not pd.api.types.is_numeric_dtype(col.dtype) and not df_is_datetime_column(df, column):
        info["n_unique"] = col.nunique()
        non_null_values = pd.Series(col[col.notnull()].unique())
        n_unique_samples = min(n_samples, len(non_null_values))
        info[f"sample"] = non_null_values.sample(n_unique_samples, random_state=42).to_list()

    return info


def data_description_main(df: pd.DataFrame) -> str:
    desc = f"DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.\n"
    desc += "index | column_name | dtype | info\n\n"
    for i, col_name in enumerate(df.columns):
        col_info = df_column_info(df, col_name)

        # Save some tokens by presenting "obvious" info without dict keys
        name = col_info.pop("field")
        json_name = json.dumps(name)
        dtype = col_info.pop("dtype")
        json_info = pd.Series(col_info).to_json(double_precision=2, date_format="iso", date_unit="s")
        col_desc = f"{i} | {json_name} | {dtype} | {json_info}"

        desc += col_desc + "\n"
    return desc


DataDescriptionStrategy = Literal["main", "head", "lida", "describe", "empty"]
DEFAULT_DATA_STRATEGY: DataDescriptionStrategy = "head"


def get_data_description_prompt(df: pd.DataFrame, description_strategy: DataDescriptionStrategy) -> str:
    match description_strategy:
        case "main":
            return data_description_main(df)
        case "head":
            return head_df_description(df)
        case "lida":
            return lida_df_description(df)
        case "describe":
            return describe_df_description(df)
        case "empty":
            return ""
        case _ as other_strategy:
            assert_never(other_strategy)
