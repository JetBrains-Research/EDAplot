import argparse
import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from benchmark.benchmark_utils import get_run_config_path
from benchmark.metrics import MetricInput, compute_vision_judge_mean
from benchmark.reports.report_utils import (
    format_path,
    get_metric_df_column_config,
    natsort_pd_index,
    sort_paths,
    st_write_raw_lc_message,
)
from benchmark.vega_chat_benchmark import get_saved_output_paths
from benchmark.vision_judge import VisionJudgeOutput
from benchmark.vision_judge_benchmark import RunConfig, VisionJudgeBenchmarkOutput, read_saved_outputs
from edaplot.image_utils import decode_image_base64


def init_page() -> None:
    st.set_page_config(page_title="EDAplot Vision Judge Benchmark Report", layout="wide")


@st.cache_resource
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, default=Path("out/vision_judge_benchmark"), help="Outputs directory")
    args = parser.parse_args()
    return args


def get_run_metadata(path: Path) -> dict[str, Any]:
    run_conf_path = get_run_config_path(path)
    run_conf = RunConfig.from_path(run_conf_path)
    meta = {
        "path": str(path),
        "inputs_path": run_conf.inputs_path,
    }
    if run_conf.vision_judge_config is not None:
        meta.update(dataclasses.asdict(run_conf.vision_judge_config))
    return meta


def compute_vision_judge_metrics(outputs: list[VisionJudgeBenchmarkOutput]) -> dict[str, float]:
    metric_inputs = []
    for out in outputs:
        metric_input = MetricInput(
            prompts=[out.input.utterance],
            messages=[],
            vision_judge_score=out.output.get_parsed_score(),
            vision_judge_is_empty_chart=out.output.get_is_empty_chart(),
        )
        metric_inputs.append(metric_input)
    return compute_vision_judge_mean(metric_inputs).to_dict("vision_judge")


@st.cache_data
def get_metrics_df_from_paths(output_paths: list[Path], names: list[str] | None = None) -> pd.DataFrame:
    all_metrics = {}
    all_meta = {}
    for i, out_path in enumerate(output_paths):
        name = names[i] if names is not None else out_path.stem
        outputs = read_saved_outputs(out_path)
        all_metrics[name] = compute_vision_judge_metrics(list(outputs.values()))
        all_meta[name] = get_run_metadata(out_path)
    individual_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    meta_df = pd.DataFrame.from_dict(all_meta, orient="index")
    return pd.concat((individual_df, meta_df), axis=1)


def write_vision_judge_output(out: VisionJudgeOutput) -> None:
    if out.score is not None:
        st.dataframe(pd.DataFrame([{"score": out.score, "label": out.label, "rationale": out.rationale}]))
    if out.criteria is not None:
        st.dataframe(pd.DataFrame([dataclasses.asdict(c) for c in out.criteria]))
    with st.container(border=True):
        col_generated, col_gt = st.columns(2)
        with col_generated:
            if out.base64_predicted is not None:
                st.image(decode_image_base64(out.base64_predicted), caption="Generated")
            else:
                st.info("No generated image available.")
        with col_gt:
            if out.base64_ground_truth is not None:
                st.image(decode_image_base64(out.base64_ground_truth), caption="Ground truth")
            else:
                st.info("No ground truth image available.")
    for judge_msg in out.messages:
        with st.chat_message("assistant"):
            st_write_raw_lc_message(judge_msg)


def write_saved_output_details(output: VisionJudgeBenchmarkOutput, key_prefix: str = "") -> None:
    st.markdown(f"### {output.input.id}")
    st.markdown(f"utterance: _{output.input.utterance}_")
    write_vision_judge_output(output.output)


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
            metrics_df[output_id] = compute_vision_judge_metrics([output])
            row_metadata = {}
            # row_metadata["dataset"] = output.input.data.dataset_name
            row_metadata["label"] = output.output.label
            row_metadata["rationale"] = output.output.rationale
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
    metrics_column_config = get_metric_df_column_config(metrics_df, False)
    column_order = metadata_columns + metrics_column_config["column_order"]
    event = st.dataframe(
        df,
        key=f"{path_str}-df",
        hide_index=False,
        selection_mode="single-row",
        on_select="rerun",
        column_order=column_order,
        column_config=metrics_column_config["column_config"],
    )
    for row_idx in event["selection"]["rows"]:
        input_id = df.index[row_idx]
        key = f"{path_str}-{input_id}"
        write_saved_output_details(outputs[input_id], key_prefix=key)


def main() -> None:
    init_page()
    cli_args = parse_cli_args()
    output_dir = cli_args.output_dir

    st.markdown("# Vision Judge Benchmark report")

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
