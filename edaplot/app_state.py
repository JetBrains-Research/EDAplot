import asyncio
from dataclasses import dataclass

import pandas as pd

from edaplot.recommend.recommender import ChartRecommender, RecommenderConfig
from edaplot.request_analyzer.header_analyzer import HeaderAnalyzer, HeaderAnalyzerConfig, HeaderAnalyzerMessage
from edaplot.request_analyzer.request_analyzer import RequestAnalyzer, RequestAnalyzerConfig, RequestAnalyzerOutput
from edaplot.vega import MessageType, SpecInfo
from edaplot.vega_chat.prompts import clear_user_prompt_formatting
from edaplot.vega_chat.vega_chat import MessageInfo, ModelConfig, VegaChat


@dataclass(kw_only=True)
class ChatMessage:
    vega_chat_message: MessageInfo
    request_analyzer_output: RequestAnalyzerOutput | None = None


class AppState:
    DATA_PROMPT_TOKEN_THRESHOLD = 4096  # About 40 tokens per column

    def __init__(self) -> None:
        self.input_df: pd.DataFrame | None = None
        self.chat: VegaChat | None = None
        self.recommender: ChartRecommender | None = None

        self.request_analyzer: RequestAnalyzer | None = None
        self.request_analyzer_enabled: bool = True

        self.header_analyzer: HeaderAnalyzer | None = None
        self.header_analyzer_messages: list[HeaderAnalyzerMessage] = []  # To show in the UI. It is run only once.

        self.n_scheduled_tasks = 0
        self.n_running_tasks = 0

        self.recommended_charts: list[SpecInfo] = []
        self.recommended_charts_selected: int | None = None
        self.used_selected_chart: SpecInfo | None = None  # last chart used by the model

        self.task_chat_data: tuple[str, MessageType, bool] | None = None
        self.task_recommender_data: int | None = None
        self.task_header_analyzer_data: bool | None = None

    @property
    def df(self) -> pd.DataFrame:
        """DataFrame used by the model, as opposed to the input DataFrame (difference in preprocessing)."""
        assert self.chat is not None
        return self.chat.dataframe

    def reset_running_state(self) -> None:
        self.n_scheduled_tasks = 0
        self.n_running_tasks = 0
        self.task_chat_data = None
        self.task_recommender_data = None
        self.task_header_analyzer_data = None

    def set_num_error_retries(self, num_retries: int) -> None:
        if self.chat is not None:
            self.chat.set_num_error_retries(num_retries)
        if self.recommender is not None:
            self.recommender.set_num_error_retries(num_retries)

    @classmethod
    def default_model_config(cls) -> ModelConfig:
        return ModelConfig(
            language="English",
            data_normalize_column_names=True,  # TODO better solution
        )

    def init_state(
        self,
        *,
        df: pd.DataFrame,
        chat_config: ModelConfig | None = None,
        n_retries: int | None = None,
        api_key: str | None = None,
    ) -> None:
        if chat_config is None:
            chat_config = self.default_model_config()

        recommender_config = RecommenderConfig(
            data_parse_dates=chat_config.data_parse_dates,
            data_normalize_column_names=chat_config.data_normalize_column_names,
        )

        if n_retries is not None:
            chat_config.n_ec_retries = n_retries
            recommender_config.n_ec_retries = n_retries

        request_analyzer_config = RequestAnalyzerConfig()
        header_analyzer_config = HeaderAnalyzerConfig()

        self.input_df = df
        self.chat = VegaChat.from_config(chat_config, self.input_df, api_key=api_key)
        self.recommender = ChartRecommender.from_config(recommender_config, self.input_df, api_key=api_key)
        self.request_analyzer = RequestAnalyzer.from_config(request_analyzer_config, self.input_df)
        self.header_analyzer = HeaderAnalyzer.from_config(header_analyzer_config)

    def reset_state(self) -> None:
        self.input_df = None
        self.chat = None
        self.recommender = None
        self.request_analyzer = None
        self.header_analyzer = None
        self.header_analyzer_messages = []
        self.recommended_charts = []
        self.recommended_charts_selected = None
        self.used_selected_chart = None
        self.reset_running_state()

    def is_busy(self) -> bool:
        return self.n_running_tasks > 0 or self.n_scheduled_tasks > 0

    def set_request_analyzer_enabled(self, enabled: bool) -> None:
        self.request_analyzer_enabled = enabled

    async def run_chat(self) -> None:
        assert self.chat is not None
        assert self.request_analyzer is not None
        assert self.task_chat_data is not None
        user_prompt, message_type, is_prompt_formatted = self.task_chat_data
        self.n_scheduled_tasks -= 1
        self.n_running_tasks += 1

        # First, notify the model of the selected chart, making sure not to repeat the same request multiple times
        selected_chart_changed = False
        if self.recommended_charts_selected is not None:
            selected_chart = self.recommended_charts[self.recommended_charts_selected]
        else:
            selected_chart = None
        if selected_chart is not None and self.used_selected_chart != selected_chart:
            self.chat.select_chart(selected_chart)
            self.used_selected_chart = selected_chart  # switching from a chart to None doesn't really make sense
            selected_chart_changed = True

        n_chat_messages = len(self.chat.session.messages)
        async with asyncio.TaskGroup() as tg:
            if self.request_analyzer_enabled and message_type == MessageType.USER:  # Only analyze real user messages
                # Add the last spec to the request analyzer's context
                last_spec = self.chat.session.last_message.spec  # N.B. could be None
                if selected_chart is not None and selected_chart_changed:
                    last_spec = selected_chart.spec
                tg.create_task(
                    self.request_analyzer.analyze_request(
                        clear_user_prompt_formatting(user_prompt) if is_prompt_formatted else user_prompt,
                        last_spec=last_spec,
                        history_idx=n_chat_messages,  # will point to the last user message
                    )
                )
            tg.create_task(self.chat.query(user_prompt, force_q=is_prompt_formatted, message_type=message_type))

        self.n_running_tasks -= 1
        self.task_chat_data = None

    async def run_recommender(self) -> None:
        assert self.recommender is not None
        assert self.task_recommender_data is not None
        self.n_scheduled_tasks -= 1
        n_charts = self.task_recommender_data
        if n_charts > 0:
            self.n_running_tasks += 1
            await self.recommender.recommend(n_charts)
            self.n_running_tasks -= 1
        self.task_recommender_data = None

    async def run_header_analyzer(self) -> None:
        assert self.header_analyzer is not None
        assert self.task_header_analyzer_data is not None
        self.n_scheduled_tasks -= 1
        self.n_running_tasks += 1
        if self.input_df is not None:
            self.header_analyzer_messages = await self.header_analyzer.analyze_header(self.input_df)
        self.n_running_tasks -= 1
        self.task_header_analyzer_data = None

    def schedule_chat(
        self, user_prompt: str, *, is_prompt_formatted: bool, message_type: MessageType = MessageType.USER
    ) -> None:
        self.n_scheduled_tasks += 1
        self.task_chat_data = (user_prompt, message_type, is_prompt_formatted)

    def schedule_recommender(self, n_charts: int) -> None:
        self.n_scheduled_tasks += 1
        self.task_recommender_data = n_charts

    def schedule_header_analyzer(self) -> None:
        self.n_scheduled_tasks += 1
        self.task_header_analyzer_data = True

    def set_recommended_charts(self, spec_infos: list[SpecInfo], selected_idx: int | None) -> None:
        self.recommended_charts = spec_infos
        self.recommended_charts_selected = selected_idx

    def undo_last_user_message(self) -> bool:
        if self.chat is None:
            return False
        user_msg_index = self.get_last_user_message_index()
        if user_msg_index is None:
            return False
        self.chat.clear_messages_from_index(user_msg_index)
        if self.request_analyzer is not None:
            self.request_analyzer.clear_messages_from_index(user_msg_index)
        return True

    def get_last_user_message_index(self) -> int | None:
        if self.chat is None:
            return -1
        return self.chat.session.get_last_user_message_index()

    def get_chat_messages(self) -> list[ChatMessage]:
        # "main" messages are from VegaChat, others are just (UI) extras
        if self.chat is None:
            return []

        messages = [ChatMessage(vega_chat_message=m) for m in self.chat.session.messages]

        # Add request analyzer messages to the final response messages so they show up in the UI
        request_analyzer_output = None
        for idx, message in enumerate(messages):
            if message.vega_chat_message.message_type == MessageType.USER:
                if idx > 0:
                    messages[idx - 1].request_analyzer_output = request_analyzer_output
                if self.request_analyzer is not None:
                    request_analyzer_output = self.request_analyzer.get_response_history(idx)
        if len(messages) > 0:
            messages[-1].request_analyzer_output = request_analyzer_output

        return messages
