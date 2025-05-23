import argparse
from pathlib import Path
from typing import Any, assert_never

import pandas as pd
import streamlit as st

from benchmark.benchmark_utils import get_run_config_path
from benchmark.evals.eval_checks import CheckRequestAnalyzer
from benchmark.evals.eval_runner import (
    RunConfig,
    check_results_to_df,
    compute_checks_metrics,
    read_saved_outputs,
    update_all_eval_checks,
)
from benchmark.evals.eval_types import ActionType, CheckType, EvalOutput
from benchmark.reports.report_utils import (
    format_path,
    get_metric_df_column_config,
    get_metric_docs,
    natsort_pd_index,
    open_vega_editor,
    sort_paths,
    st_vega_lite_chart,
    st_write_codeblock,
    st_write_raw_lc_message,
)
from benchmark.vega_chat_benchmark import get_saved_output_paths
from edaplot.data_utils import spec_remove_data
from edaplot.request_analyzer.header_analyzer import get_header_analyzer_warning
from edaplot.spec_utils import SpecType
from edaplot.vega import MessageType, VegaMessage

# These message types are shown optionally
DEBUG_MESSAGE_TYPES = [MessageType.SYSTEM, MessageType.USER_ERROR_CORRECTION, MessageType.AI_RESPONSE_ERROR]


def init_page() -> None:
    st.set_page_config(page_title="EDAplot Evals Report", layout="wide")


@st.cache_resource
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="Outputs directory")
    args = parser.parse_args()
    return args


def write_response_vega_message(message: VegaMessage, df: pd.DataFrame | None) -> None:
    is_message_parsed = message.explanation is not None or len(message.spec_infos) > 0
    with st.chat_message("assistant"):
        with st.expander("See raw LLM response"):
            st_write_raw_lc_message(message.message)
        if message.explanation is not None:
            st.write(message.explanation)
        for spec_info in message.spec_infos:
            st.json(spec_info.spec, expanded=True)
            st_vega_lite_chart(spec_info.spec, df)
        if not is_message_parsed:
            # Raw LLM response
            st.write(message.message.content)


def write_vega_message(message: VegaMessage, df: pd.DataFrame | None) -> None:
    match message.message_type:
        case MessageType.USER:
            with st.chat_message("user"):
                st.write(message.message.content)
        case MessageType.USER_ERROR_CORRECTION:
            with st.chat_message("user"):
                st.write(message.message.content)
        case MessageType.SYSTEM | MessageType.AI_RESPONSE_VALID | MessageType.AI_RESPONSE_ERROR:
            write_response_vega_message(message, df)
        case _ as unreachable:  # See https://docs.python.org/3.11/library/typing.html#typing.assert_never
            assert_never(unreachable)


def should_show_message(message: VegaMessage, show_all_messages: bool) -> bool:
    return show_all_messages or message.message_type not in DEBUG_MESSAGE_TYPES


def get_run_metadata(path: Path) -> dict[str, Any]:
    run_conf_path = get_run_config_path(path)
    run_conf = RunConfig.from_path(run_conf_path)
    meta = {
        "chat_model_name": run_conf.chat_model_config.model_name,
        "path": str(path),
    }
    return meta


@st.cache_data
def get_metrics_df_from_paths(output_paths: list[Path], names: list[str] | None = None) -> pd.DataFrame:
    all_metrics = {}
    all_meta = {}
    for i, out_path in enumerate(output_paths):
        outputs = read_saved_outputs(out_path)
        all_check_results = []
        for out in outputs.values():
            all_check_results.extend(out.check_results)
        name = names[i] if names is not None else out_path.stem
        all_metrics[name] = compute_checks_metrics(all_check_results)
        all_meta[name] = get_run_metadata(out_path)
    individual_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    meta_df = pd.DataFrame.from_dict(all_meta, orient="index")
    return pd.concat((individual_df, meta_df), axis=1)


def write_saved_output_details(output: EvalOutput, key_prefix: str = "") -> None:
    st.markdown(f"### {output.input.id}")
    data_df = output.input.load_dataframe()
    results_summary_df = check_results_to_df(output.check_results)
    results_summary_column_config = get_metric_df_column_config(results_summary_df, False)
    st.dataframe(results_summary_df, **results_summary_column_config)
    for i, (action, action_output, check_results) in enumerate(
        zip(output.input.actions, output.action_outputs, output.check_results)
    ):
        st.write(f"Step {i}: {action.action_type}")
        wkey = f"{key_prefix}-{i}-{action.action_type}"
        match action.action_type:
            case ActionType.USER_UTTERANCE:
                with st.container(border=True):
                    st_write_codeblock(action.action_kwargs["user_utterance"])
                    st.write("**Generated**")

                    if (
                        request_analyzer_output := CheckRequestAnalyzer.get_request_analyzer_response([action_output])
                    ) is not None:
                        st.markdown("**Request analyzer response**")
                        if request_analyzer_output.request_type is not None:
                            st.write(request_analyzer_output.request_type[-1].response)
                        if request_analyzer_output.data_availability is not None:
                            st.write(request_analyzer_output.data_availability[-1].response)

                    if (
                        action_output.vega_chat_messages is None
                        or len(action_output.vega_chat_messages) == 0
                        or action_output.vega_chat_messages[-1].spec is None
                    ):
                        st.warning("No Vega-Lite specs generated")
                    else:
                        cur_message = action_output.vega_chat_messages[-1]
                        with st.container(border=True):
                            st_vega_lite_chart(cur_message.spec, data_df)
                        if st.button("Open in Vega Editor", key=f"{wkey}-generated_vega_editor"):
                            # Opening the editor this way passes the data to the editor :)
                            assert cur_message.spec is not None
                            open_vega_editor(cur_message.spec, data_df)
                        st.json(cur_message.spec, expanded=False)

                # st.write(check_results)
                gt_specs: list[SpecType] = list(
                    *(
                        check.check_kwargs.get("specs", [])
                        for check in action.checks
                        if check.check_type == CheckType.GROUND_TRUTH
                    )
                )
                if len(gt_specs) > 0:
                    with st.container(border=True):
                        st.write("**Ground Truth**")
                        for gt_spec in gt_specs:
                            spec_remove_data(gt_spec)
                            with st.container(border=True):
                                st_vega_lite_chart(gt_spec, data_df)
                            if st.button("Open in Vega Editor", key=f"{wkey}_gt_vega_editor"):
                                open_vega_editor(gt_spec, data_df)
                            st.json(gt_spec, expanded=False)
                else:
                    st.info("No Ground Truth check available")
            case ActionType.SELECT_CHART:
                with st.container(border=True):
                    st.write("**Selected chart**")
                    spec = action.action_kwargs["spec"]
                    spec_remove_data(spec)
                    with st.container(border=True):
                        st_vega_lite_chart(spec, data_df)
                    if st.button("Open in Vega Editor", key=f"{wkey}_vega_editor"):
                        open_vega_editor(spec, data_df)
                    st.json(spec, expanded=False)
            case ActionType.HEADER_ANALYZER:
                with st.container(border=True):
                    st.write("**Header analyzer**")
                    if action_output.header_analyzer_history is not None:
                        response = action_output.header_analyzer_history[-1].response
                        if response is not None:
                            warning_needed, warning_msg = get_header_analyzer_warning(response)
                            if warning_needed:
                                st.warning(warning_msg)
                            else:
                                st.info("No warnings emitted")
                        with st.expander("See dataset header analysis"):
                            st.write("**Dataset header analysis**")
                            for header_analyzer_msg in action_output.header_analyzer_history:
                                with st.chat_message("assistant"):
                                    st_write_codeblock(header_analyzer_msg.message.content)
                    else:
                        st.warning("No header analyzer history")
            case _ as unreachable:
                assert_never(unreachable)

        with st.expander("See checks"):
            for check in action.checks:
                st.write(f"**{check.check_type}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Input")
                    st.write(check.check_kwargs)
                with col2:
                    st.write("Output")
                    st.write(check_results[check.check_type])

    with st.container(border=True):
        if st.checkbox("Show dataset", key=f"{key_prefix}-show_dataset", disabled=data_df is None):
            st.dataframe(data_df, key=f"{key_prefix}-dataset")
        if st.checkbox("Show chat", value=False, key=f"{key_prefix}-show_chat"):
            show_all_messages = st.checkbox(
                "Show all messages",
                value=True,
                key=f"{key_prefix}-show_all_messages",
            )
            st.divider()
            vega_messages = [
                m.to_vega_message()
                for action_output in output.action_outputs
                if action_output.vega_chat_messages is not None
                for m in action_output.vega_chat_messages
            ]
            request_analyzer_outputs = [
                m
                for action_output in output.action_outputs
                if action_output.request_analyzer_history is not None
                for m in action_output.request_analyzer_history
            ]
            for vega_message, request_analyzer_output in zip(vega_messages, request_analyzer_outputs):
                if request_analyzer_output is not None:
                    if show_all_messages:
                        with st.expander("See request analyzer response"):
                            st.markdown("**Request analyzer response**")
                            if request_analyzer_output.request_type is not None:
                                st.markdown("**Request type**")
                                for request_type_msg in request_analyzer_output.request_type:
                                    st_write_codeblock(request_type_msg.message.content)
                            if request_analyzer_output.data_availability is not None:
                                st.markdown("**Data availability**")
                                for data_availability_msg in request_analyzer_output.data_availability:
                                    st_write_codeblock(data_availability_msg.message.content)
                    else:
                        if request_analyzer_output.data_availability is not None:
                            st_write_codeblock(request_analyzer_output.data_availability[-1].message.content)
                        if request_analyzer_output.request_type is not None:
                            st_write_codeblock(request_analyzer_output.request_type[-1].message.content)
                if should_show_message(vega_message, show_all_messages):
                    write_vega_message(vega_message, data_df)


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
            metrics_df[output_id] = compute_checks_metrics(output.check_results)
            row_metadata = {}
            row_metadata["dataset"] = output.input.data.dataset_name
            row_metadata["data_id"] = output.input.data.dataset_item_id or output.input.data.path
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


def update_saved_eval_checks(outputs_path: Path) -> None:
    with st.sidebar:
        with st.spinner(text="Updating check results..."):
            update_all_eval_checks(outputs_path)
            st.cache_data.clear()


def main() -> None:
    init_page()
    cli_args = parse_cli_args()
    output_dir = cli_args.output_dir

    st.markdown("# Evals report")

    with st.sidebar:
        st.button(
            "Update check results",
            help="Update the _stored_ check results for every file",
            on_click=lambda: update_saved_eval_checks(output_dir),
        )

    docs = get_metric_docs()
    with st.expander("See metric definitions"):
        for metric_name, doc in docs.items():
            st.markdown(f"- **{metric_name}**")
            st.markdown(doc)

    all_paths = sort_paths(get_saved_output_paths(output_dir))
    all_names = [format_path(p, output_dir) for p in all_paths]
    df = get_metrics_df_from_paths(all_paths, all_names)
    df_column_config = get_metric_df_column_config(df, False)

    st.write("Select a row to view detailed results for the file.")
    event = st.dataframe(
        df,
        **df_column_config,
        selection_mode="single-row",
        on_select="rerun",
    )
    for row_idx in event["selection"]["rows"]:
        st.write(f"## {all_names[row_idx]}")
        write_saved_outputs(all_paths[row_idx])


if __name__ == "__main__":
    main()
