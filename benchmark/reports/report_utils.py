import inspect
import re
from pathlib import Path
from typing import Any, Iterable

import altair as alt
import pandas as pd
import streamlit as st
from langchain_core.messages import BaseMessage

from benchmark.metrics import (
    compute_spec_score,
    empty_plot_error_rate,
    lida_sevq_score,
    response_error_rate,
    spec_f1_correctness_encoding,
    spec_f1_correctness_full,
    spec_f1_correctness_key,
    spec_f1_correctness_key_values,
    spec_f1_correctness_mark,
    spec_f1_correctness_transform,
    spec_jaccard_keys,
    visualization_error_rate,
)
from edaplot.spec_utils import SpecType
from edaplot.vega import to_altair_chart


def remove_ci_metrics(metrics: dict[str, float], nan_only: bool) -> dict[str, float]:
    """Remove confidence intervals from metrics."""

    def should_keep(k: str, v: float) -> bool:
        if k.endswith("_ci_low") or k.endswith("_ci_high"):
            return not nan_only or not pd.isna(v)
        return True

    return {k: v for k, v in metrics.items() if should_keep(k, v)}


def natsort_key(s: str) -> list[str | int]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def sort_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: natsort_key(p.name))


def natsort_pd_index(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    def _natsort(index: pd.Index) -> pd.Index:
        # Sort ids (strings) as integers if possible
        return index.map(natsort_key)

    df.sort_index(key=_natsort, inplace=inplace)
    return df


@st.cache_data
def get_metric_docs() -> dict[str, str | None]:
    docs = {
        "spec_score": compute_spec_score,
        "lida_sevq": lida_sevq_score,
        "visualization_error_rate": visualization_error_rate,
        "response_error_rate": response_error_rate,
        "mark_f1": spec_f1_correctness_mark,
        "encoding_f1": spec_f1_correctness_encoding,
        "transform_f1": spec_f1_correctness_transform,
        "full_f1": spec_f1_correctness_full,
        "kvs_f1": spec_f1_correctness_key_values,
        "keys_f1": spec_f1_correctness_key,
        "keys_jaccard": spec_jaccard_keys,
        "empty_plot_rate": empty_plot_error_rate,
    }
    docs = {k: inspect.getdoc(fn) for k, fn in docs.items()}
    return docs


def get_metric_df_column_config(df: pd.DataFrame, show_f1_pr: bool) -> dict[str, Any]:
    docs = get_metric_docs()
    visible_columns = []
    for c in df.columns:
        if not show_f1_pr and (c.endswith("_precision") or c.endswith("_recall")):
            continue
        visible_columns.append(c)
    column_config = {k: st.column_config.NumberColumn(help=doc, width="small") for k, doc in docs.items()}
    # Sort columns so that those starting with "check_" are first
    check_columns = [col for col in visible_columns if col.startswith("check_")]
    other_columns = [col for col in visible_columns if not col.startswith("check_")]
    sorted_columns = sorted(check_columns) + other_columns
    return {"column_config": column_config, "column_order": sorted_columns}


def format_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix(""))


def st_vega_lite_chart(
    spec: SpecType | None,
    df: pd.DataFrame | None,
    use_container_width: bool = False,
) -> None:
    if df is None:
        st.warning("🚨 Couldn't find the spec's data. Make sure you are using the correct path to the dataset.")
    else:
        st.vega_lite_chart(df, spec, use_container_width=use_container_width)


def st_write_raw_lc_message(message: BaseMessage) -> None:
    st.write(message.model_dump())


def st_write_codeblock(content: str | list) -> None:
    assert isinstance(content, str)  # for convenient mypy checks
    st.code(content, language=None)


def open_vega_editor(spec: SpecType, df: pd.DataFrame) -> None:
    # Opening the editor can still fail sometimes...
    # Sub-sample because: `OSError: [Errno 7] Argument list too long: 'firefox'`
    n = min(len(df), 500)
    sub_df = alt.sample(df, n=n)
    chart = to_altair_chart(spec, sub_df)
    chart.open_editor()
