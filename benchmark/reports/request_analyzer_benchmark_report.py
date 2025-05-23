import argparse
import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from benchmark.benchmark_utils import get_run_config_path
from benchmark.reports.report_utils import format_path, natsort_pd_index, sort_paths, st_write_codeblock
from benchmark.request_analyzer_benchmark import (
    RequestAnalyzerBenchmarkOutput,
    RunConfig,
    compute_request_analyzer_benchmark_metrics,
    read_saved_outputs,
)
from benchmark.vega_chat_benchmark import get_saved_output_paths
from edaplot.request_analyzer.request_analyzer import get_request_analyzer_warning


def init_page() -> None:
    st.set_page_config(page_title="EDAplot Request Analyzer Benchmark Report", layout="wide")


@st.cache_resource
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=Path, nargs="?", default=Path("out/request_analyzer_benchmark"), help="Outputs directory"
    )
    args = parser.parse_args()
    return args


def get_run_metadata(path: Path) -> dict[str, Any]:
    run_conf_path = get_run_config_path(path)
    run_conf = RunConfig.from_path(run_conf_path)
    meta = {
        "path": str(path),
        "dataset": run_conf.dataset_config["name"],
        "request_analyzer_config": dataclasses.asdict(run_conf.request_analyzer_config),
    }
    return meta


@st.cache_data
def get_metrics_df_from_paths(output_paths: list[Path], names: list[str] | None = None) -> pd.DataFrame:
    all_metrics = {}
    all_meta = {}
    for i, out_path in enumerate(output_paths):
        name = names[i] if names is not None else out_path.stem
        outputs = read_saved_outputs(out_path)
        metrics = compute_request_analyzer_benchmark_metrics(list(outputs.values()))
        all_metrics[name] = {k: v for d in metrics.values() for k, v in d.items()}
        all_metrics[name]["count"] = len(outputs)
        all_meta[name] = get_run_metadata(out_path)
    individual_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    meta_df = pd.DataFrame.from_dict(all_meta, orient="index")
    return pd.concat((meta_df, individual_df), axis=1)


def write_request_analyzer_benchmark_output(output: RequestAnalyzerBenchmarkOutput) -> None:
    st.markdown(f"**{output.input.id}**")

    for prompt, response in zip(output.input.prompt, output.response_history):
        st_write_codeblock(prompt)
        with st.expander("See request analyzer response"):
            st.markdown("**Request analyzer response**")
            if response.request_type is not None:
                st.markdown("**Request type**")
                for request_type_msg in response.request_type:
                    st_write_codeblock(request_type_msg.message.content)
            if response.data_availability is not None:
                st.markdown("**Data availability**")
                for data_availability_msg in response.data_availability:
                    st_write_codeblock(data_availability_msg.message.content)

        if response.request_type is not None:
            st.write(response.request_type[-1].response)
        if response.data_availability is not None:
            st.write(response.data_availability[-1].response)

        warning_needed, warning_msg = get_request_analyzer_warning(response)
        if warning_needed:
            st.warning(warning_msg)
        else:
            st.info("No warning emitted.")

    with st.container(border=True):
        st.write("**Ground truth**")
        if output.input.ground_truth is not None:
            st.json(output.input.ground_truth, expanded=False)
        else:
            st.info("No ground truth available.")


def write_saved_outputs(outputs_file: Path) -> None:
    path_str = str(outputs_file)
    if path_str in st.session_state:
        outputs = st.session_state[path_str]["outputs"]
        metrics_df = st.session_state[path_str]["metrics_df"]
        metadata_df = st.session_state[path_str]["metadata_df"]
    else:
        outputs = read_saved_outputs(outputs_file)
        metrics_df = {}
        metadata_df = {}
        for output_id, output in outputs.items():
            metrics = compute_request_analyzer_benchmark_metrics([output])
            metrics_df[output_id] = {"all": metrics["all"]["all"]}
            row_metadata = {}
            row_metadata["benchmark_type"] = output.input.benchmark_type.value
            row_metadata["dropped_fields"] = output.dropped_fields
            metadata_df[output_id] = row_metadata
        metrics_df = pd.DataFrame(metrics_df).T
        metadata_df = pd.DataFrame(metadata_df).T
        st.session_state[path_str] = {"outputs": outputs, "metrics_df": metrics_df, "metadata_df": metadata_df}

    df = pd.concat((metadata_df, metrics_df), axis=1)
    df.index.rename("id", inplace=True)
    df = natsort_pd_index(df)
    metrics_columns = metrics_df.columns.to_list()
    metadata_columns = metadata_df.columns.to_list()

    st.write("Select a row to view the example's details.")
    column_order = metadata_columns + metrics_columns
    event = st.dataframe(
        df,
        key=f"{path_str}-df",
        hide_index=False,
        selection_mode="single-row",
        on_select="rerun",
        column_order=column_order,
    )
    for row_idx in event["selection"]["rows"]:
        input_id = df.index[row_idx]
        write_request_analyzer_benchmark_output(outputs[input_id])


def main() -> None:
    init_page()
    cli_args = parse_cli_args()
    output_dir = cli_args.output_dir

    st.markdown("# Request Analyzer Benchmark report")

    all_paths = sort_paths(get_saved_output_paths(output_dir))
    all_names = [format_path(p, output_dir) for p in all_paths]
    df = get_metrics_df_from_paths(all_paths, all_names)

    st.write("Select a row to view detailed results for the file.")
    event = st.dataframe(
        df,
        selection_mode="single-row",
        on_select="rerun",
    )
    for row_idx in event["selection"]["rows"]:
        st.write(f"## {all_names[row_idx]}")
        write_saved_outputs(all_paths[row_idx])


if __name__ == "__main__":
    main()
