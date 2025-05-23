from dataclasses import dataclass
from logging import getLogger
from typing import Self

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, trim_messages
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from edaplot.data_prompts import DEFAULT_DATA_STRATEGY, DataDescriptionStrategy
from edaplot.data_utils import df_preprocess
from edaplot.recommend.prompts import (
    extract_model_response,
    get_multiple_replies_prompt,
    get_system_prompt,
    get_user_prompt,
)
from edaplot.vega import MessageType, SpecInfo, process_extracted_specs, validate_and_fix_spec
from edaplot.vega_chat.prompts import get_error_correction_prompt

logger = getLogger(__name__)


@dataclass
class RecommenderMessage:
    message: BaseMessage
    message_type: MessageType
    spec_infos: list[SpecInfo]
    explanation: str | None = None


@dataclass(kw_only=True)
class RecommenderConfig:
    model_name: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.0

    n_ec_retries: int = 1
    description_strategy: DataDescriptionStrategy = DEFAULT_DATA_STRATEGY
    message_trimmer_max_tokens: int = 8192
    retry_on_empty_plot: bool = True

    data_normalize_column_names: bool = False
    data_parse_dates: bool = True


class ChartRecommender:
    def __init__(
        self,
        config: RecommenderConfig,
        df: pd.DataFrame,
        chat_model: BaseChatModel,
    ) -> None:
        self.config = config
        self._df = df_preprocess(
            df, normalize_column_names=config.data_normalize_column_names, parse_dates=config.data_parse_dates
        )
        self._chat = chat_model
        self._n_retries = config.n_ec_retries
        self._max_error_length = int(config.message_trimmer_max_tokens * 0.33)
        self._message_trimmer = trim_messages(
            max_tokens=config.message_trimmer_max_tokens,
            strategy="last",
            token_counter=self._chat,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        self._is_running = False  # To help the UI
        system_prompt = get_system_prompt(self._df, data_description_strategy=self.config.description_strategy)
        self._messages: list[RecommenderMessage] = [self.create_message(system_prompt, MessageType.SYSTEM)]

    def set_num_error_retries(self, new_value: int) -> None:
        self._n_retries = new_value

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def messages(self) -> list[RecommenderMessage]:
        return self._messages

    @property
    def last_message(self) -> RecommenderMessage:
        # There is always at least the system message
        return self._messages[-1]

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @staticmethod
    def create_message(content: str, message_type: MessageType) -> RecommenderMessage:
        message = MessageType.create_message(content, message_type)
        return RecommenderMessage(message, message_type, [])

    async def _invoke(self, content: str, message_type: MessageType) -> RecommenderMessage:
        message = self.create_message(content, message_type)
        self._messages.append(message)

        messages = [m.message for m in self._messages]
        messages = self._message_trimmer.invoke(messages)
        response = await self._chat.ainvoke(messages)
        logger.info(response)

        response_message = RecommenderMessage(response, MessageType.AI_RESPONSE_VALID, [])
        self._messages.append(response_message)
        return response_message

    def process_response(self, response: RecommenderMessage) -> tuple[RecommenderMessage, str | None]:
        assert isinstance(response.message.content, str)

        reply: str | None = None
        try:
            extracted_response = extract_model_response(response.message.content, allow_multiple=True)
        except ValueError as e:
            reply = get_error_correction_prompt(str(e), max_length=self._max_error_length)
            response.message_type = MessageType.AI_RESPONSE_ERROR
            return response, reply

        replies = []
        spec_infos = []
        for raw_spec in extracted_response.specs:
            extracted_spec = process_extracted_specs([raw_spec])
            spec_fix = validate_and_fix_spec(
                extracted_spec,
                self._df,
                retry_on_empty_plot=self.config.retry_on_empty_plot,
                max_reply_length=self._max_error_length,
            )
            assert spec_fix.spec_validity is not None

            spec_info = SpecInfo(
                spec=spec_fix.spec,
                is_spec_fixed=spec_fix.spec != extracted_spec,
                is_empty_chart=spec_fix.spec_validity.is_empty_scenegraph,
                is_valid_schema=spec_fix.spec_validity.is_valid_schema,
                is_drawable=spec_fix.spec_validity.is_valid_scenegraph,
            )
            spec_infos.append(spec_info)
            replies.append(spec_fix.reply)

        response.explanation = extracted_response.explanation
        response.spec_infos = spec_infos

        has_reply = any(reply is not None for reply in replies)
        if has_reply:
            reply = get_multiple_replies_prompt(replies)

        any_not_drawable = any(not spec.is_drawable for spec in spec_infos)
        any_is_empty = self.config.retry_on_empty_plot and any(spec.is_empty_chart for spec in spec_infos)
        if any_not_drawable or any_is_empty:
            assert reply is not None
            response.message_type = MessageType.AI_RESPONSE_ERROR

        return response, reply

    async def recommend(self, n_charts: int) -> RecommenderMessage:
        self._is_running = True
        content: str | None = get_user_prompt(n_charts)
        n_attempts = 0
        while n_attempts <= self._n_retries and content is not None:
            msg_type = MessageType.USER_ERROR_CORRECTION if n_attempts > 0 else MessageType.USER
            response = await self._invoke(content, msg_type)
            response, content = self.process_response(response)
            n_attempts += 1
        self._is_running = False
        return self.last_message

    def gather_all_charts(self, include_invalid: bool = False, include_empty: bool = False) -> list[SpecInfo]:
        spec_infos = []
        for message in self.messages:
            for spec_info in message.spec_infos:
                if include_invalid or spec_info.is_drawable:
                    if include_invalid or include_empty or not spec_info.is_empty_chart:
                        spec_infos.append(spec_info)
        return spec_infos

    @classmethod
    def from_config(cls, config: RecommenderConfig, df: pd.DataFrame, api_key: str | None = None) -> Self:
        chat_model = get_chat_model_from_config(config, api_key=api_key)
        return cls(
            config,
            df,
            chat_model,
        )


def get_chat_model_from_config(config: RecommenderConfig, api_key: str | None = None) -> BaseChatModel:
    secret_api_key = SecretStr(api_key) if api_key else None
    return ChatOpenAI(model=config.model_name, temperature=config.temperature, api_key=secret_api_key)
