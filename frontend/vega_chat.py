import asyncio
import logging
import os
import time
from typing import assert_never

import pandas as pd
import streamlit as st
import vega_datasets
from streamlit.runtime.uploaded_file_manager import UploadedFile

from edaplot.api import make_interactive_spec
from edaplot.app_state import AppState, ChatMessage
from edaplot.data_prompts import get_data_description_prompt
from edaplot.recommend.recommender import RecommenderMessage
from edaplot.request_analyzer.header_analyzer import HeaderAnalyzerMessage, get_header_analyzer_warning
from edaplot.request_analyzer.request_analyzer import RequestAnalyzerOutput, get_request_analyzer_warning
from edaplot.spec_utils import SpecType
from edaplot.vega import MessageType, SpecInfo
from edaplot.vega_chat.prompts import (
    clear_user_prompt_formatting,
    get_new_chart_recommendation_user_prompt,
    get_user_prompt,
)
from edaplot.vega_chat.vega_chat import ChatSession
from edaplot.vega_chat.vega_chat import get_chat_model_from_config as get_vega_chat_model_from_config
from frontend.plotting import PlottingFrontend, st_vega_lite_chart

logger = logging.getLogger(__name__)

_prefix = "vega_chat"

KEY_STATE = f"{_prefix}_state"

WKEY_NUM_EC_RETRIES = f"{_prefix}_w_num_ec_retries"
WKEY_DATASET_SELECTION = f"{_prefix}_w_dataset_selection"
WKEY_N_CHARTS = f"{_prefix}_w_n_charts"
WKEY_OPENAI_KEY = f"{_prefix}_w_openai_key"
WKEY_SHOW_ALL_MESSAGES = f"{_prefix}_w_show_all_messages"
WKEY_SHOW_DF = f"{_prefix}_w_show_df"
WKEY_SELECTED_RECOMMENDED_CHART = f"{_prefix}_w_selected_recommended_chart"
WKEY_PLOTTING_FRONTEND = f"{_prefix}_w_plotting_frontend"
WKEY_INTERACTIVE_CHARTS = f"{_prefix}_w_interactive_charts"
WKEY_RUN_HEADER_ANALYSIS = f"{_prefix}_w_run_header_analysis"
WKEY_RUN_REQUEST_ANALYSIS = f"{_prefix}_w_run_request_analysis"
WKEY_REPORT_LETS_PLOT_CONVERTER_SUMMARY = f"{_prefix}_w_report_lets_plot_converter_summary"
WKEY_LOG_LETS_PLOT_SPEC = f"{_prefix}_w_log_lets_plot_spec"

SUPPORTED_DF_FILE_TYPES = ["csv", "json", "xlsx", "tsv"]


def init_page() -> None:
    st.set_page_config(page_title="EDAplot Chat", layout="wide")


def init_or_get_state() -> AppState:
    if KEY_STATE not in st.session_state:
        st.session_state[KEY_STATE] = AppState()
    return st.session_state[KEY_STATE]


def get_state() -> AppState:
    return st.session_state[KEY_STATE]


def get_openai_api_key() -> str:
    return st.session_state[WKEY_OPENAI_KEY]


def is_debug_mode() -> bool:
    return st.session_state[WKEY_SHOW_ALL_MESSAGES]


def st_write_codeblock(content: str | list) -> None:
    assert isinstance(content, str)  # for convenient mypy checks
    st.code(content, language=None)


def load_dataframe(dataset_name: str) -> pd.DataFrame:
    return vega_datasets.data(dataset_name)


def check_dataframe_exceeds_limits(df: pd.DataFrame) -> bool:
    # We only need to know the model that will be used to get its tokenizer for checking limits.
    # We currently use the same model and data prompt for all models.
    chat_config = AppState.default_model_config()
    chat_model = get_vega_chat_model_from_config(chat_config)
    data_prompt = get_data_description_prompt(df, chat_config.description_strategy)
    num_tokens = chat_model.get_num_tokens(data_prompt)
    if num_tokens > AppState.DATA_PROMPT_TOKEN_THRESHOLD:
        return True
    return False


def load_uploaded_file_dataframe(
    uploaded_file: UploadedFile, encoding: str = "utf-8", max_rows: int | None = None
) -> pd.DataFrame:
    # TODO use a common loading function everywhere
    file_type = uploaded_file.name.split(".")[-1]

    # From lida utils
    read_funcs = {
        "csv": lambda: pd.read_csv(uploaded_file, encoding=encoding),
        "json": lambda: pd.read_json(uploaded_file, orient="records", encoding=encoding),
        "xlsx": lambda: pd.read_excel(uploaded_file),
        "tsv": lambda: pd.read_csv(uploaded_file, sep="\t", encoding=encoding),
    }
    assert set(SUPPORTED_DF_FILE_TYPES) == set(read_funcs.keys())
    if file_type not in read_funcs:
        raise ValueError(f"Unsupported file type: {file_type}")

    df = read_funcs[file_type]()
    if max_rows is not None and len(df) > max_rows:
        logger.info(f"Sampling df to max={max_rows} rows...")
        df = df.sample(max_rows)
    return df


def _make_interactive_spec(df: pd.DataFrame | None, spec: SpecType | None) -> SpecType | None:
    if spec is None or df is None:
        return spec
    if st.session_state[WKEY_INTERACTIVE_CHARTS]:
        return make_interactive_spec(df, spec)
    return spec


def write_recommended_charts(df: pd.DataFrame, spec_infos: list[SpecInfo]) -> list[str]:
    n_charts = len(spec_infos)
    chart_labels = ["None"]
    if n_charts == 0:
        st.error("Got 0 charts.")
    else:
        st_cols = st.columns(n_charts)
        for i, spec_info in enumerate(spec_infos):
            chart_label = f"Chart **#{i + 1}**"
            chart_labels.append(chart_label)
            with st_cols[i]:
                st.markdown(chart_label)
                spec = _make_interactive_spec(df, spec_info.spec)
                if spec is not None:
                    with st.expander("See vega-lite specification"):
                        st.json(spec, expanded=True)
                if spec_info.is_drawable:
                    plotting_frontend = st.session_state[WKEY_PLOTTING_FRONTEND]
                    report_lets_plot_converter_summary = st.session_state[WKEY_REPORT_LETS_PLOT_CONVERTER_SUMMARY]
                    log_lets_plot_spec = st.session_state[WKEY_LOG_LETS_PLOT_SPEC]
                    st_vega_lite_chart(
                        spec,
                        df,
                        use_container_width=True,
                        plotting_frontend=plotting_frontend,
                        report_lets_plot_converter_summary=report_lets_plot_converter_summary,
                        log_lets_plot_spec=log_lets_plot_spec,
                    )
                elif spec is not None:
                    st.warning("🚨 Invalid spec: failed to show plot.")
    return chart_labels


def write_recommended_charts_selectable(df: pd.DataFrame, spec_infos: list[SpecInfo]) -> None:
    chart_labels = write_recommended_charts(df, spec_infos)

    state = get_state()
    selected_chart_label = st.radio(
        "Select a chart to chat with",
        chart_labels,
        horizontal=True,
        key=WKEY_SELECTED_RECOMMENDED_CHART,
        disabled=state.is_busy(),
    )
    selected_idx = chart_labels.index(selected_chart_label)
    state.set_recommended_charts(spec_infos, None if selected_idx == 0 else selected_idx - 1)


def write_recommender_response(message: RecommenderMessage, df: pd.DataFrame) -> None:
    with st.chat_message("assistant"):
        if message.explanation is not None:
            with st.expander("See AI explanation"):
                st.write(message.explanation)
        write_recommended_charts(df, message.spec_infos)


def write_recommender_message(message: RecommenderMessage, df: pd.DataFrame) -> None:
    match message.message_type:
        case MessageType.USER | MessageType.USER_ERROR_CORRECTION:
            with st.chat_message("user"):
                st.write(message.message.content)
        case MessageType.AI_RESPONSE_VALID | MessageType.AI_RESPONSE_ERROR:
            write_recommender_response(message, df)
        case MessageType.SYSTEM:
            with st.chat_message("assistant"):
                st_write_codeblock(message.message.content)
        case _ as unreachable:  # See https://docs.python.org/3.11/library/typing.html#typing.assert_never
            assert_never(unreachable)


def write_request_analyzer_output(output: RequestAnalyzerOutput) -> None:
    if is_debug_mode():
        with st.expander("See request analyzer response"):
            st.markdown("**Request analyzer response**")
            if output.request_type is not None:
                st.markdown("**Request type**")
                for request_type_msg in output.request_type:
                    st_write_codeblock(request_type_msg.message.content)
            if output.data_availability is not None:
                st.markdown("**Data availability**")
                for data_availability_msg in output.data_availability:
                    st_write_codeblock(data_availability_msg.message.content)
    warning_needed, warning_msg = get_request_analyzer_warning(output)
    if warning_needed:
        st.warning(warning_msg)


def write_chat_response(message: ChatMessage, df: pd.DataFrame | None) -> None:
    vega_msg = message.vega_chat_message
    spec = _make_interactive_spec(df, vega_msg.spec)
    with st.chat_message("assistant"):
        if is_debug_mode():
            with st.expander("See raw response"):
                st_write_codeblock(vega_msg.message.content)

        if message.request_analyzer_output is not None:
            write_request_analyzer_output(message.request_analyzer_output)

        vega_response = vega_msg.model_response
        if vega_response is not None:
            if vega_response.relevant_request == False:  # None != False
                st.warning(
                    f"We detected that your request may not be relevant to the input dataset. _AI: {vega_response.relevant_request_rationale}_"
                )
            if vega_response.data_exists == False:
                st.warning(
                    f"We detected that the input dataset might not contain the required data. _AI: {vega_response.data_exists_rationale}_"
                )
            # with st.expander("See AI explanation"):
            st.write(vega_response.explanation)

        if spec is not None:
            with st.expander("See vega-lite specification"):
                st.json(spec, expanded=True)
        if vega_msg.is_drawable:
            plotting_frontend = st.session_state[WKEY_PLOTTING_FRONTEND]
            st_vega_lite_chart(spec, df, use_container_width=True, plotting_frontend=plotting_frontend)
        elif spec is not None:
            st.warning("🚨 Invalid spec: failed to show plot.")
        if vega_response is None:
            st.write(vega_msg.message.content)


def write_chat_user_message(message: ChatMessage) -> None:
    content = message.vega_chat_message.message.content
    assert isinstance(content, str)
    with st.chat_message("user"):
        st.write(clear_user_prompt_formatting(content))


def write_chat_user_message_undoable(message: ChatMessage) -> None:
    state = get_state()
    cols = st.columns([10, 1], vertical_alignment="center")  # Move
    with cols[0].container():
        write_chat_user_message(message)
    with cols[1]:
        undo_clicked = st.button("Undo", disabled=state.is_busy(), use_container_width=True, type="primary")
        if undo_clicked and state.undo_last_user_message():
            st.rerun()


def write_chat_message(message: ChatMessage, df: pd.DataFrame | None) -> None:
    vega_msg = message.vega_chat_message
    match vega_msg.message_type:
        case MessageType.USER:
            write_chat_user_message(message)
        case MessageType.USER_ERROR_CORRECTION:
            with st.chat_message("user"):
                st.write(vega_msg.message.content)
        case MessageType.AI_RESPONSE_VALID | MessageType.AI_RESPONSE_ERROR:
            write_chat_response(message, df)
        case MessageType.SYSTEM:
            with st.chat_message("assistant"):
                st_write_codeblock(vega_msg.message.content)
        case _ as unreachable:  # See https://docs.python.org/3.11/library/typing.html#typing.assert_never
            assert_never(unreachable)


def should_show_chat_message(message: ChatMessage, is_last_msg: bool) -> bool:
    if is_debug_mode():
        return True
    msg_type = message.vega_chat_message.message_type
    if msg_type == MessageType.AI_RESPONSE_ERROR:
        return is_last_msg
    return msg_type not in [
        MessageType.SYSTEM,
        MessageType.USER_ERROR_CORRECTION,
        MessageType.AI_RESPONSE_ERROR,
    ]


def should_show_recommender_message(message: RecommenderMessage) -> bool:
    return is_debug_mode() or message.message_type not in [
        MessageType.USER,
        MessageType.SYSTEM,
        MessageType.USER_ERROR_CORRECTION,
        MessageType.AI_RESPONSE_ERROR,
    ]


def write_all_recommender_messages() -> None:
    state = get_state()
    assert state.recommender is not None
    recommender_only_sys_msg = len(state.recommender.messages) == 1
    if is_debug_mode():
        for rec_message in state.recommender.messages:
            if should_show_recommender_message(rec_message):
                write_recommender_message(rec_message, state.df)
        st.divider()
    # Show all recommended charts (might be scattered across messages due to error retries)
    if not recommender_only_sys_msg and not state.recommender.is_running:
        recommended_spec_infos = state.recommender.gather_all_charts()
        write_recommended_charts_selectable(state.df, recommended_spec_infos)


def write_all_chat_messages() -> None:
    state = get_state()
    assert state.chat is not None
    undoable_user_msg_index = state.get_last_user_message_index()
    messages = state.get_chat_messages()
    for i, chat_message in enumerate(messages):
        is_last_msg = i == len(messages) - 1
        if should_show_chat_message(chat_message, is_last_msg):
            if i == undoable_user_msg_index:
                write_chat_user_message_undoable(chat_message)
            else:
                write_chat_message(chat_message, state.df)


def write_header_analysis(messages: list[HeaderAnalyzerMessage]) -> None:
    if len(messages) == 0:
        return
    if is_debug_mode():
        with st.expander("See dataset header analysis"):
            st.write("**Dataset header analysis**")
            for message in messages:
                with st.chat_message("assistant"):
                    st_write_codeblock(message.message.content)
    response_msg = messages[-1]
    if response_msg.response is not None:
        warning_needed, warning_msg = get_header_analyzer_warning(response_msg.response)
        if warning_needed:
            with st.container(border=True):
                st.markdown("**Dataset header analysis**")
                st.warning(warning_msg)


def update_num_ec_retries() -> None:
    num_retries = st.session_state[WKEY_NUM_EC_RETRIES]
    get_state().set_num_error_retries(num_retries)


def write_dataset(df: pd.DataFrame | None, name: str | None) -> None:
    if df is None:
        st.error("No dataset selected.")
        return
    if name is not None:
        st.markdown(f"Dataset **{repr(name)}**:")
    st.dataframe(df)


def main() -> None:
    init_page()
    state = init_or_get_state()

    with st.sidebar:
        st.header("Vega-Lite Chat")

        if WKEY_SHOW_ALL_MESSAGES not in st.session_state:
            st.session_state[WKEY_SHOW_ALL_MESSAGES] = False
        st.checkbox(
            "Show all messages",
            help="Show _all_ messages exchanged by the user and the LLM (including the **system prompt** and "
            "error correction messages).",
            key=WKEY_SHOW_ALL_MESSAGES,
        )

        if WKEY_RUN_HEADER_ANALYSIS not in st.session_state:
            st.session_state[WKEY_RUN_HEADER_ANALYSIS] = True
        analyze_dataset_headers = st.checkbox(
            "Analyze dataset header",
            key=WKEY_RUN_HEADER_ANALYSIS,
            help="Analyze dataset column headers for clarity when loading a new dataset.",
            disabled=state.is_busy(),
        )

        if WKEY_RUN_REQUEST_ANALYSIS not in st.session_state:
            st.session_state[WKEY_RUN_REQUEST_ANALYSIS] = True
        st.checkbox(
            "Analyze user requests",
            key=WKEY_RUN_REQUEST_ANALYSIS,
            help="Analyze user requests for clarity and feasibility and display warnings if necessary.",
            disabled=state.is_busy(),
            on_change=lambda: state.set_request_analyzer_enabled(st.session_state[WKEY_RUN_REQUEST_ANALYSIS]),
        )

        if WKEY_N_CHARTS not in st.session_state:
            st.session_state[WKEY_N_CHARTS] = 3
        n_charts = st.slider(
            "Initial charts to recommend",
            min_value=0,
            max_value=5,
            help="Show this many recommended charts when selecting a dataset. Select **0** to disable automatic recommendations.",
            disabled=state.is_busy(),
            key=WKEY_N_CHARTS,
        )

        recommend_btn_pressed = st.button(
            "Recommend chart in Chat",
            help="Recommend a _single_ chart in the context of the current chat.",
            disabled=state.is_busy(),
        )

        st.subheader("Model configuration")
        env_key = os.environ.get("OPENAI_API_KEY")
        if WKEY_OPENAI_KEY not in st.session_state:
            st.session_state[WKEY_OPENAI_KEY] = env_key
        if get_openai_api_key() is None:
            st.text_input(
                "OpenAI API Key",
                type="password",
                on_change=state.reset_state,
                disabled=state.is_busy(),
                key=WKEY_OPENAI_KEY,
            )
        if WKEY_NUM_EC_RETRIES not in st.session_state:
            st.session_state[WKEY_NUM_EC_RETRIES] = 3
        num_ec_retries = st.slider(
            "Error correction retries",
            min_value=0,
            max_value=5,
            key=WKEY_NUM_EC_RETRIES,
            help="If the LLM outputs an invalid schema, automatically try to correct the error.",
            on_change=update_num_ec_retries,
            disabled=state.is_busy(),
        )

        st.subheader("Plotting")
        if WKEY_PLOTTING_FRONTEND not in st.session_state:
            st.session_state[WKEY_PLOTTING_FRONTEND] = PlottingFrontend.VEGA_LITE
        st.selectbox(
            "Plotting frontend (experimental)",
            [PlottingFrontend.VEGA_LITE, PlottingFrontend.LETS_PLOT, PlottingFrontend.LETS_PLOT_VEGA_LITE],
            help="Select the plotting frontend to use for rendering the plots.",
            key=WKEY_PLOTTING_FRONTEND,
            format_func=lambda x: {
                PlottingFrontend.LETS_PLOT: "Lets-Plot",
                PlottingFrontend.VEGA_LITE: "Vega-Lite",
                PlottingFrontend.LETS_PLOT_VEGA_LITE: "Lets-Plot + Vega-Lite",
            }[x],
        )

        if WKEY_INTERACTIVE_CHARTS not in st.session_state:
            st.session_state[WKEY_INTERACTIVE_CHARTS] = True
        st.checkbox(
            "Interactive charts",
            key=WKEY_INTERACTIVE_CHARTS,
            help="Make charts interactive (e.g., zoom, pan, tooltips).",
        )

        if WKEY_REPORT_LETS_PLOT_CONVERTER_SUMMARY not in st.session_state:
            st.session_state[WKEY_REPORT_LETS_PLOT_CONVERTER_SUMMARY] = False

        st.checkbox(
            "Report converter summary",
            help="Report the summary of the Lets-Plot converter using COMPUTATION_MESSAGES.",
            key=WKEY_REPORT_LETS_PLOT_CONVERTER_SUMMARY,
        )

        if WKEY_LOG_LETS_PLOT_SPEC not in st.session_state:
            st.session_state[WKEY_LOG_LETS_PLOT_SPEC] = False

        st.checkbox(
            "Log Lets-Plot spec",
            help="Print plotSpec to a console.",
            key=WKEY_LOG_LETS_PLOT_SPEC,
        )

    if get_openai_api_key() is None:
        st.warning("Please add your OpenAI API key to continue.", icon="🚨")
        st.stop()

    st.write("Upload your _tabular_ dataset or pick a sample dataset.")
    dataset_cols = st.columns(3, vertical_alignment="bottom")
    with dataset_cols[0]:
        uploaded_file = st.file_uploader(
            "Upload your dataset", type=SUPPORTED_DF_FILE_TYPES, on_change=state.reset_state
        )
    with dataset_cols[1]:
        _datasets_list = vega_datasets.local_data.list_datasets()
        if WKEY_DATASET_SELECTION not in st.session_state:
            st.session_state[WKEY_DATASET_SELECTION] = None
        dataset_selection: str | None = st.selectbox(
            "Pick a dataset",
            _datasets_list,
            on_change=state.reset_state,
            disabled=state.is_busy() or uploaded_file is not None,
            key=WKEY_DATASET_SELECTION,
        )
    with dataset_cols[2]:
        should_show_df = st.checkbox("Show dataset", key=WKEY_SHOW_DF)

    # Pinned to the bottom of the page
    user_input = st.chat_input("Say something", disabled=state.is_busy() or state.input_df is None)

    if dataset_selection is None and uploaded_file is None:
        st.error("No dataset selected.")
        st.stop()
    elif state.input_df is None:
        if uploaded_file is not None:
            df = load_uploaded_file_dataframe(uploaded_file, max_rows=4500)
        elif dataset_selection is not None:
            df = load_dataframe(dataset_selection)
        else:
            st.error("No dataset selected.")
            st.stop()

        if check_dataframe_exceeds_limits(df):
            st.error(
                f"The selected dataset exceeds the allowed number of tokens ({AppState.DATA_PROMPT_TOKEN_THRESHOLD}). "
                f"Please select a smaller dataset or reduce the number of columns in the dataset."
            )
            st.stop()

        state.init_state(
            api_key=get_openai_api_key(),
            df=df,
            n_retries=num_ec_retries,
        )
        st.rerun()  # enable disabled widgets

    # State is initialized. Start the main UI.
    assert state.input_df is not None
    assert state.chat is not None
    assert state.recommender is not None
    assert state.header_analyzer is not None

    # Draw main UI
    with st.container(border=True):
        if should_show_df:
            name = uploaded_file.name if uploaded_file is not None else dataset_selection
            write_dataset(state.input_df, name)

        if len(state.header_analyzer_messages) > 0:
            write_header_analysis(state.header_analyzer_messages)

    chat_only_sys_msg = len(state.chat.session.messages) == 1
    recommender_only_sys_msg = len(state.recommender.messages) == 1

    container_recommended_charts = st.container(border=True)
    if n_charts > 0 or not recommender_only_sys_msg:
        with container_recommended_charts:
            st.markdown("Recommended charts")
            write_all_recommender_messages()

    chat_container = st.container(border=True)
    with chat_container:
        st.markdown("Chat")
        write_all_chat_messages()

    # Run actions after drawing the main UI
    # N.B. All "actions" should be below widgets for the widgets' states to be saved!
    should_rerun = False
    if state.n_scheduled_tasks == 0 and state.n_running_tasks == 0:  # Waiting on user input
        if (
            analyze_dataset_headers
            and len(state.header_analyzer_messages) == 0
            and state.task_header_analyzer_data is None
        ):
            # Run header analysis when the dataset is first loaded and the setting is enabled
            state.schedule_header_analyzer()
            should_rerun = True
        if recommend_btn_pressed:
            state.schedule_chat(
                get_new_chart_recommendation_user_prompt(), message_type=MessageType.USER, is_prompt_formatted=True
            )
            should_rerun = True
        if chat_only_sys_msg and recommender_only_sys_msg and n_charts > 0:
            state.schedule_recommender(n_charts=n_charts)
            should_rerun = True
        if user_input:
            state.schedule_chat(get_user_prompt(user_input), message_type=MessageType.USER, is_prompt_formatted=True)
            should_rerun = True
    elif state.n_scheduled_tasks > 0:  # Scheduled tasks

        async def run_scheduled_tasks() -> None:
            # Run multiple tasks in parallel
            async with asyncio.TaskGroup() as tg:
                tasks = []
                if state.task_chat_data is not None:
                    with chat_container:
                        # Immediately display the user's message
                        user_prompt, message_type, _ = state.task_chat_data
                        user_message = ChatSession.create_message(user_prompt, message_type)
                        user_message = ChatMessage(vega_chat_message=user_message, request_analyzer_output=None)
                        if should_show_chat_message(user_message, True):
                            write_chat_message(user_message, state.df)
                    tasks.append(tg.create_task(state.run_chat()))
                if state.task_header_analyzer_data is not None:
                    tasks.append(tg.create_task(state.run_header_analyzer()))
                if state.task_recommender_data is not None:
                    tasks.append(tg.create_task(state.run_recommender()))
                # TODO figure out how to update the UI when a task completes... (st.rerun() doesn't work)

        with st.spinner("Generating response..."):
            asyncio.run(run_scheduled_tasks())
        should_rerun = True
    elif state.n_running_tasks > 0:
        # This state happens when the "scheduled" state is interrupted by the user interacting with the UI.
        with st.spinner("Generating response..."):
            while state.n_running_tasks > 0:
                time.sleep(0.2)
        should_rerun = True

    # Rerun to notify widgets about the new running_state
    # N.B. Make sure to call this below all widgets!
    if should_rerun:
        st.rerun()


if __name__ == "__main__":
    main()
