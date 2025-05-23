import argparse
import dataclasses
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from benchmark.benchmark_utils import get_run_config_path
from benchmark.datasets import DatasetItem
from benchmark.metrics import compute_metrics
from benchmark.models.base_models import EvalMessage, EvalModelType
from benchmark.models.eval_models import preprocess_eval_model_df
from benchmark.models.lida import LIDAModelConfig
from benchmark.models.vega_chat import VegaChatEvalConfig
from benchmark.reports.evals_report import write_vega_message
from benchmark.reports.report_utils import (
    format_path,
    get_metric_df_column_config,
    get_metric_docs,
    natsort_pd_index,
    open_vega_editor,
    remove_ci_metrics,
    sort_paths,
    st_vega_lite_chart,
    st_write_codeblock,
)
from benchmark.reports.vision_judge_benchmark_report import write_vision_judge_output
from benchmark.vega_chat_benchmark import (
    RunConfig,
    SavedOutput,
    get_saved_output_paths,
    load_dataset_from_outputs_path,
    read_saved_outputs,
)
from edaplot.data_utils import spec_remove_data
from edaplot.image_utils import decode_image_base64
from edaplot.spec_utils import SpecType, get_spec_marks
from edaplot.vega import MessageType, SpecInfo

# These message types are shown optionally
DEBUG_MESSAGE_TYPES = [MessageType.SYSTEM, MessageType.USER_ERROR_CORRECTION, MessageType.AI_RESPONSE_ERROR]

KEY_METRIC_SHOW_F1_PR = "metric_show_f1_pr"
KEY_INCLUDE_GEOSHAPE = "exclude_geoshape"
KEY_SAVED_OUTPUTS = "saved_outputs"


def init_page() -> None:
    st.set_page_config(page_title="EDAplot Evaluation Report", layout="wide")


def init_session_state() -> None:
    st.session_state[KEY_SAVED_OUTPUTS] = {}


def reset_session_state() -> None:
    if KEY_SAVED_OUTPUTS in st.session_state:
        del st.session_state[KEY_SAVED_OUTPUTS]
    st.cache_data.clear()


@st.cache_resource
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="Outputs directory")
    args = parser.parse_args()
    return args


def should_show_message(message: EvalMessage, show_all_messages: bool) -> bool:
    return show_all_messages or message.message_type not in DEBUG_MESSAGE_TYPES


def should_skip_output(output: SavedOutput) -> bool:
    if not st.session_state[KEY_INCLUDE_GEOSHAPE] and (gt_spec := output.ground_truth) is not None:
        return "geoshape" in get_spec_marks(gt_spec)
    return False


def get_run_metadata(path: Path) -> dict[str, Any]:
    run_conf_path = get_run_config_path(path)
    run_conf = RunConfig.from_path(run_conf_path)
    meta = {
        "_dataset_name": run_conf.dataset_config["name"],
        "_path": str(path),
    }
    if isinstance(run_conf.model_config, VegaChatEvalConfig):
        meta["_model_name"] = run_conf.model_config.model_config.model_name
    elif isinstance(run_conf.model_config, LIDAModelConfig):
        meta["_model_name"] = run_conf.model_config.model_name
    return meta


@st.cache_data
def get_metrics_df_from_paths(output_paths: list[Path], names: list[str] | None = None) -> pd.DataFrame:
    all_metric_inputs = []
    all_metrics = {}
    all_meta = {}
    for i, out_path in enumerate(output_paths):
        outputs = read_saved_outputs(out_path)
        metric_inputs = [out.to_metric_input() for out in outputs if not should_skip_output(out)]
        metrics = compute_metrics(metric_inputs)
        name = names[i] if names is not None else out_path.stem
        all_metrics[name] = metrics
        all_meta[name] = get_run_metadata(out_path)
        all_metric_inputs.extend(metric_inputs)
    individual_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    meta_df = pd.DataFrame.from_dict(all_meta, orient="index")
    all_df = pd.DataFrame()
    all_df["all_macro"] = individual_df.mean(axis=0)
    all_df["all_micro"] = compute_metrics(all_metric_inputs)
    df = pd.concat((pd.concat((individual_df, meta_df), axis=1), all_df.T), axis=0)
    # No sense in averaging confidence intervals
    df.loc["all_macro", df.columns.str.endswith("_ci_low") | df.columns.str.endswith("_ci_high")] = float("nan")
    return df


def write_metric_chart(metrics_df: pd.DataFrame, metric_name: str) -> None:
    cols = ["path", metric_name, f"{metric_name}_ci_low", f"{metric_name}_ci_high", "count"]
    cols += [c for c in metrics_df.columns if c.startswith("_")]
    chart_df = metrics_df.reset_index(names="path").filter(cols)
    chart_df["version"] = chart_df["path"].map(lambda s: s.split("/")[0])
    # chart_df["filename"] = chart_df["path"].map(lambda s: s.split("/")[-1])
    base = alt.Chart(chart_df).encode(
        y=alt.Y("path"),
    )
    bars = base.mark_bar().encode(
        x=alt.X(metric_name, title=metric_name),
        tooltip=chart_df.columns.tolist(),
        color="version",
    )
    error_bars = base.mark_errorbar(ticks=True).encode(
        x=alt.X(f"{metric_name}_ci_low", title=metric_name),
        x2=f"{metric_name}_ci_high",
    )
    base_chart = bars + error_bars
    # Faceting doesn't work properly with streamlit: https://github.com/streamlit/streamlit/issues/9091
    chart = alt.vconcat().resolve_scale(x="shared")
    for ds_name in chart_df["_dataset_name"].unique():
        if not pd.isna(ds_name):
            chart &= base_chart.properties(title=ds_name).transform_filter(f'datum._dataset_name == "{ds_name}"')
    st.altair_chart(chart, use_container_width=True)


def write_generated_vl_spec(spec: SpecType | None, df: pd.DataFrame | None, key_prefix: str = "") -> None:
    if spec is not None:
        with st.container(border=True):
            st_vega_lite_chart(spec, df)
    else:
        st.warning("🚨 Invalid schema: failed to show plot.")
    can_open_chart = spec is not None and df is not None  # and output.final_message.is_valid_schema
    if st.button("Open in Vega Editor", key=f"{key_prefix}-generated_vega_editor", disabled=not can_open_chart):
        # Opening the editor this way passes the data to the editor :)
        assert spec is not None and df is not None
        open_vega_editor(spec, df)
    st.json(spec, expanded=False)


def write_saved_output_details(
    output: SavedOutput, dataset_item: DatasetItem | None, run_config: RunConfig, *, key_prefix: str = ""
) -> None:
    st.markdown(f"### {output.id}")

    original_data_df = dataset_item.data if dataset_item is not None else None
    processed_data_df = original_data_df
    if original_data_df is not None:
        processed_data_df = preprocess_eval_model_df(original_data_df, run_config.model_config)
    for prompt in output.prompt:
        st_write_codeblock(prompt)

    with st.container(border=True):
        st.write("**Generated**")
        final_message = output.final_message
        if final_message.model_type == EvalModelType.VEGA_CHAT:
            # For backwards compatibility (before we had model_type)
            lida_library = final_message.message.additional_kwargs.get("library")
            if lida_library is not None:
                with st.expander(f"See _{lida_library}_ code:"):
                    st.code(final_message.message.additional_kwargs.get("code"))
            write_generated_vl_spec(final_message.spec, processed_data_df, key_prefix=key_prefix)
        elif final_message.model_type == EvalModelType.LIDA:
            lida_library = final_message.message.additional_kwargs.get("library")
            with st.expander(f"See _{lida_library}_ code:"):
                st.code(final_message.code)
            if final_message.spec is not None:
                write_generated_vl_spec(final_message.spec, processed_data_df, key_prefix=key_prefix)
            if final_message.base64_raster is not None:
                st.image(decode_image_base64(final_message.base64_raster))
        elif final_message.model_type == EvalModelType.CoML4Vis:
            with st.expander(f"See code:"):
                st.code(final_message.code)
            if final_message.explanation is not None:
                st.write("Error:")
                st.code(final_message.explanation)
            if final_message.base64_raster is not None:
                st.image(decode_image_base64(final_message.base64_raster))
        else:
            raise NotImplementedError(final_message.model_type)

    with st.container(border=True):
        st.write("**Ground truth**")
        if output.ground_truth is not None:
            gt_spec = output.ground_truth
            spec_remove_data(gt_spec)
            with st.container(border=True):
                st_vega_lite_chart(gt_spec, original_data_df)
            can_open_chart = original_data_df is not None
            if st.button("Open in Vega Editor", key=f"{key_prefix}-gt_vega_editor", disabled=not can_open_chart):
                assert original_data_df is not None
                open_vega_editor(gt_spec, original_data_df)
            st.json(gt_spec, expanded=False)
        else:
            st.info("No ground truth available.")

    with st.expander("Metrics"):
        single_metrics = compute_metrics([output.to_metric_input()])
        metrics_df = pd.DataFrame([single_metrics])
        st.dataframe(metrics_df, key=f"{key_prefix}-metrics")

        # Show lida self eval details (full LLM response)
        if output.lida_self_eval is not None:
            st.markdown(f"**LIDA self evaluation** - {single_metrics['lida_sevq']:.2f}")
            st.write(dataclasses.asdict(output.lida_self_eval))

        if output.vision_judge is not None:
            st.markdown(f"**Vision judge evaluation**")
            write_vision_judge_output(output.vision_judge)

    with st.container(border=True):
        if st.checkbox("Show dataset", key=f"{key_prefix}-show_dataset", disabled=dataset_item is None):
            st.dataframe(original_data_df, key=f"{key_prefix}-dataset")
        if st.checkbox("Show chat", value=False, key=f"{key_prefix}-show_chat"):
            show_all_messages = st.checkbox(
                "Show all messages",
                value=True,
                help="Show _all_ messages exchanged by the user and the LLM (including the **system prompt** and "
                "error correction messages).",
                key=f"{key_prefix}-show_all_messages",
            )
            st.divider()
            for message in output.messages:
                if should_show_message(message, show_all_messages):
                    write_vega_message(message.to_vega_message(), processed_data_df)


def write_saved_outputs(outputs_file: Path) -> None:
    path_str = str(outputs_file)
    if path_str in st.session_state[KEY_SAVED_OUTPUTS]:
        cached_outputs = st.session_state[KEY_SAVED_OUTPUTS][path_str]
        outputs = cached_outputs["outputs"]
        metrics_df = cached_outputs["metrics_df"]
        metadata_df = cached_outputs["metadata_df"]
        run_config = cached_outputs["run_config"]
    else:
        outputs = read_saved_outputs(outputs_file)
        outputs = {output.id: output for output in outputs}
        run_config_path = get_run_config_path(outputs_file)
        run_config = RunConfig.from_path(run_config_path)
        try:
            dataset = load_dataset_from_outputs_path(outputs_file)
        except FileNotFoundError:
            dataset = None
        metrics_df = {}
        metadata_df = {}
        for output_id, output in outputs.items():
            if should_skip_output(output):
                continue
            row_metrics = compute_metrics([output.to_metric_input()])
            row_metrics.pop("count")  # Always == 1
            row_metrics = remove_ci_metrics(row_metrics, nan_only=True)
            metrics_df[output.id] = row_metrics
            row_metadata = {}
            if dataset is not None:
                data = dataset[output_id]
                if (metadata := data.metadata) is not None:
                    row_metadata.update(metadata)
                spec_info = (
                    output.final_message.spec_infos[0]
                    if len(output.final_message.spec_infos) > 0
                    else SpecInfo(spec={})
                )
                row_metadata["is_drawable"] = spec_info.is_drawable
                row_metadata["is_empty_chart"] = spec_info.is_empty_chart
                row_metadata["is_valid_schema"] = spec_info.is_valid_schema
                row_metadata["is_spec_fixed"] = spec_info.is_spec_fixed
            if (gt_spec := output.ground_truth) is not None:
                gt_marks = "-".join(sorted(set(get_spec_marks(gt_spec))))
                row_metadata["gt_mark"] = gt_marks
            metadata_df[output_id] = row_metadata
        metrics_df = pd.DataFrame(metrics_df).T
        metadata_df = pd.DataFrame(metadata_df).T
        st.session_state[KEY_SAVED_OUTPUTS][path_str] = {
            "outputs": outputs,
            "metrics_df": metrics_df,
            "metadata_df": metadata_df,
            "run_config": run_config,
        }

    df = pd.concat((metadata_df, metrics_df), axis=1)
    df.index.rename("id", inplace=True)
    df = natsort_pd_index(df)
    metrics_columns = metrics_df.columns.to_list()
    metadata_columns = metadata_df.columns.to_list()

    group_by_metadata: str = st.selectbox(
        "Group by:",
        metadata_columns + ["describe"],
        key=f"{path_str}-group_by_metadata",
        help="Show aggregated metrics (mean) grouped by the specified metadata field.",
    )
    if group_by_metadata != "describe":
        # TODO fix group_by for e.g. token usage stats
        group_by_df = df.groupby(group_by_metadata)[metrics_columns].mean()
        group_by_df.insert(0, "num", df.groupby(group_by_metadata).size())
    else:
        group_by_df = df.describe(include="all")
        group_by_df.insert(0, "num", len(df))
    group_by_column_config = get_metric_df_column_config(group_by_df, st.session_state[KEY_METRIC_SHOW_F1_PR])
    st.dataframe(group_by_df, **group_by_column_config, key=f"{path_str}-group_by_df")

    st.text("Select a row to view the example's details.")
    metrics_column_config = get_metric_df_column_config(metrics_df, st.session_state[KEY_METRIC_SHOW_F1_PR])
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
    if len(event["selection"]["rows"]) > 0:
        try:
            dataset = load_dataset_from_outputs_path(outputs_file)
        except FileNotFoundError as e:
            dataset = None
            st.exception(e)

        for row_idx in event["selection"]["rows"]:
            id_ = df.index[row_idx]
            dataset_item = dataset[id_] if dataset is not None else None
            key = f"{path_str}-{id_}"
            write_saved_output_details(outputs[id_], dataset_item, run_config, key_prefix=key)


def main() -> None:
    init_page()
    init_session_state()
    cli_args = parse_cli_args()
    output_dir = cli_args.output_dir

    st.markdown("# Benchmarks report")

    docs = get_metric_docs()
    with st.expander("See metric definitions"):
        for metric_name, doc in docs.items():
            st.markdown(f"- **{metric_name}**")
            st.markdown(doc)

    show_f1_col, show_chart_col, show_tree_col, show_geoshape_col = st.columns(4)
    with show_f1_col:
        st.checkbox(
            "Show F1 _precision_ and _recall_ metrics",
            value=False,
            help="Show the precision and recall used to compute the F1/F_beta scores.",
            key=KEY_METRIC_SHOW_F1_PR,
        )
    with show_chart_col:
        should_show_chart = st.checkbox("Show charts summary", value=True)
    with show_tree_col:
        should_show_tree = st.checkbox(
            "Show directory summary", help="Group results by the directory structure.", value=False
        )
    with show_geoshape_col:
        st.checkbox(
            "Include _geoshape_ mark types",
            value=False,
            help="Whether to exclude ground truth examples containing the geoshape mark type, as we currently don't support it.",
            key=KEY_INCLUDE_GEOSHAPE,
            on_change=reset_session_state,
        )

    all_paths = get_saved_output_paths(output_dir)
    all_paths = sort_paths(all_paths)
    all_names = [format_path(p, output_dir) for p in all_paths]
    df = get_metrics_df_from_paths(all_paths, all_names)
    df_column_config = get_metric_df_column_config(df, st.session_state[KEY_METRIC_SHOW_F1_PR])

    df.to_csv(output_dir / "summary.csv")

    st.write("Select a row to view detailed results for the file.")
    event = st.dataframe(
        df,
        **df_column_config,
        selection_mode="single-row",
        on_select="rerun",
    )

    def write_tree(dir_path: Path, level: int) -> None:
        st.markdown(f"{'#' * level} {dir_path.name}")
        outputs_here = get_saved_output_paths(dir_path, recursive=False)
        if len(outputs_here) > 0:
            df = get_metrics_df_from_paths(outputs_here)
            df_column_config = get_metric_df_column_config(df, st.session_state[KEY_METRIC_SHOW_F1_PR])
            st.dataframe(df, **df_column_config)
        for path in sort_paths(dir_path.iterdir()):
            if path.is_dir():
                write_tree(path, level + 1)

    if should_show_chart:
        write_metric_chart(df, "spec_score")

    if should_show_tree:
        write_tree(output_dir, 2)

    for row_idx in event["selection"]["rows"]:
        st.write(f"## {all_names[row_idx]}")
        write_saved_outputs(all_paths[row_idx])


if __name__ == "__main__":
    main()
