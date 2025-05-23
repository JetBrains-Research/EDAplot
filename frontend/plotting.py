import copy
import enum

import pandas as pd
import streamlit as st
from streamlit_lets_plot import lets_plot_chart

from edaplot.spec_utils import SpecType


class PlottingFrontend(enum.Enum):
    VEGA_LITE = (True, False)
    LETS_PLOT = (False, True)
    LETS_PLOT_VEGA_LITE = (True, True)

    def __init__(self, vega_lite: bool, lets_plot: bool) -> None:
        self.vega_lite = vega_lite
        self.lets_plot = lets_plot


def st_vega_lite_chart(
    spec: SpecType | None,
    df: pd.DataFrame | None,
    use_container_width: bool = False,
    plotting_frontend: PlottingFrontend = PlottingFrontend.VEGA_LITE,
    report_lets_plot_converter_summary: bool = False,
    log_lets_plot_spec: bool = False,
) -> None:
    if df is None:
        st.warning("🚨 Couldn't find the spec's data. Make sure you are using the correct path to the dataset.")
        return

    if plotting_frontend.vega_lite:
        st.vega_lite_chart(df, spec, use_container_width=use_container_width)

    if plotting_frontend.lets_plot:
        from altair import to_values
        from lets_plot._type_utils import standardize_dict  # type: ignore

        # Convert datetime to int (and other types to expected by lets-plot)
        # Calling to_values() directly  produces incompatible types, e.g. str for datetime.
        df = standardize_dict(df)  # converts types to expected by lets-plot (datetime to int)
        assert isinstance(df, dict)  # for mypy

        # Convert tabular format to values format (list of dicts)
        values = to_values(pd.DataFrame.from_dict(df))

        spec_with_data = copy.deepcopy(spec)
        spec_with_data["reportLetsPlotConverterSummary"] = report_lets_plot_converter_summary
        spec_with_data["logLetsPlotSpec"] = log_lets_plot_spec
        spec_with_data["data"] = values
        lets_plot_chart(spec_with_data, use_container_width=True)
