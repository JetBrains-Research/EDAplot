import dataclasses
from dataclasses import dataclass
from typing import Any, Self

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, trim_messages
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from edaplot.data_prompts import DEFAULT_DATA_STRATEGY, DataDescriptionStrategy
from edaplot.data_utils import df_preprocess
from edaplot.spec_utils import SpecType, spec_is_empty
from edaplot.vega import (
    MessageType,
    SpecInfo,
    VegaMessage,
    append_reply,
    logger,
    make_text_spec,
    process_extracted_specs,
    validate_and_fix_spec,
)
from edaplot.vega_chat.prompts import (
    ModelResponse,
    PromptVersion,
    extract_model_response,
    get_error_correction_prompt,
    get_select_spec_info_prompt,
    get_spec_fixed_user_prompt,
    get_system_prompt,
    get_user_prompt,
)


@dataclass(kw_only=True)
class MessageInfo:
    # TODO replace with VegaMessage
    # defaults for an invalid response
    # Store the chart validity because it's a slow operation to recompute each time.
    message: BaseMessage
    message_type: MessageType
    spec: SpecType | None = None
    is_spec_fixed: bool = False
    is_empty_chart: bool = True
    is_valid_schema: bool = False
    is_drawable: bool = False
    model_response: ModelResponse | None = None

    def get_spec_info(self) -> SpecInfo | None:
        if self.spec is None:
            return None
        return SpecInfo(
            spec=self.spec,
            is_spec_fixed=self.is_spec_fixed,
            is_empty_chart=self.is_empty_chart,
            is_valid_schema=self.is_valid_schema,
            is_drawable=self.is_drawable,
        )

    def to_vega_message(self) -> VegaMessage:
        spec_info = self.get_spec_info()
        return VegaMessage(
            message=self.message,
            message_type=self.message_type,
            spec_infos=[] if spec_info is None else [spec_info],
            explanation=self.model_response.explanation if self.model_response is not None else None,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        d["message_type"] = MessageType(d["message_type"])
        if "model_response" in d:
            d["model_response"] = ModelResponse(**d["model_response"]) if d["model_response"] is not None else None
        else:
            d["model_response"] = ModelResponse(specs=[], explanation=d.pop("explanation"))
        return cls(**d)


@dataclass(kw_only=True)
class ModelConfig:
    model_name: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.0

    language: str | None = "English"
    n_ec_retries: int = 3
    description_strategy: DataDescriptionStrategy = DEFAULT_DATA_STRATEGY
    message_trimmer_max_tokens: int = 8192
    retry_on_empty_plot: bool = True
    prompt_version: PromptVersion = "vega_chat_v1"
    retry_on_irrelevant_request: bool = False

    # TODO normalizing column names breaks comparison with ground truth specs!
    data_normalize_column_names: bool = False
    data_parse_dates: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        fields = {f.name for f in dataclasses.fields(cls)}
        extra_fields = set()
        for k in d:
            if k not in fields:
                logger.warning(f"Skipping unknown config field: {k}")
                extra_fields.add(k)
        return cls(**{k: v for k, v in d.items() if k not in extra_fields})


class ChatSession:
    def __init__(
        self,
        chat: BaseChatModel,
        system_prompt: str,
        *,
        trimmer_max_tokens: int = 8192,
    ):
        self._chat = chat
        self._messages: list[MessageInfo] = [self.create_message(system_prompt, MessageType.SYSTEM)]
        self._message_trimmer = trim_messages(
            max_tokens=trimmer_max_tokens,
            strategy="last",
            token_counter=self._chat,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

    @property
    def messages(self) -> list[MessageInfo]:
        return self._messages

    @property
    def last_message(self) -> MessageInfo:
        # There is always at least the system message
        return self._messages[-1]

    def get_last_user_message_index(self) -> int | None:
        for i, m in enumerate(reversed(self._messages)):
            if m.message_type == MessageType.USER:
                return len(self._messages) - i - 1
        return None

    @staticmethod
    def create_message(content: str, message_type: MessageType) -> MessageInfo:
        message = MessageType.create_message(content, message_type)
        return MessageInfo(message=message, message_type=message_type)

    def add_message(self, content: str, message_type: MessageType) -> MessageInfo:
        message = self.create_message(content, message_type)
        self._messages.append(message)
        return message

    def clear_messages_from_index(self, message_index: int) -> None:
        if message_index <= 0:
            raise ValueError("message_index must be > 0")
        self._messages = self._messages[:message_index]

    async def invoke(self, content: str, message_type: MessageType) -> MessageInfo:
        self.add_message(content, message_type)

        messages = [m.message for m in self._messages]
        messages = self._message_trimmer.invoke(messages)
        response = await self._chat.ainvoke(messages)
        logger.info(response)

        response_message = MessageInfo(message=response, message_type=MessageType.AI_RESPONSE_VALID)
        self._messages.append(response_message)
        return response_message


class VegaChat:
    def __init__(
        self,
        config: ModelConfig,
        df: pd.DataFrame,
        chat_model: BaseChatModel,
        metadata: str = "",
    ) -> None:
        self.config = config  # Read-only config used to initialize the System
        self._df = df_preprocess(
            df, normalize_column_names=config.data_normalize_column_names, parse_dates=config.data_parse_dates
        )

        system_prompt = get_system_prompt(
            self.config.prompt_version,
            df=self._df,
            data_description_strategy=config.description_strategy,
            extra_metadata=metadata,
            language=config.language,
        )
        self._session = ChatSession(
            chat_model,
            system_prompt,
            trimmer_max_tokens=config.message_trimmer_max_tokens,
        )
        self._n_retries = config.n_ec_retries
        self._is_running = False  # To help the UI
        self._max_error_length = int(config.message_trimmer_max_tokens * 0.33)
        self._spec_history: dict[int, SpecInfo] = {}  # Map from message index to spec

    def set_num_error_retries(self, new_value: int) -> None:
        self._n_retries = new_value

    @property
    def session(self) -> ChatSession:
        return self._session

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    def get_user_prompt(self, content: str) -> str:
        # For multi-turn prompts, tell the LLM about the possibly fixed spec so it doesn't use
        # the incorrect generated one
        last_ai_message = self.session.last_message
        if last_ai_message.is_spec_fixed:
            assert last_ai_message.spec is not None
            return get_spec_fixed_user_prompt(last_ai_message.spec, content)
        return get_user_prompt(content)

    def add_user_message(self, content: str, message_type: MessageType = MessageType.USER) -> MessageInfo:
        return self.session.add_message(content, message_type)

    def process_response(self, response: MessageInfo) -> tuple[MessageInfo, str | None]:
        assert isinstance(response.message.content, str)

        try:
            extracted_response = extract_model_response(response.message.content)
            extracted_spec = process_extracted_specs(extracted_response.specs)
        except ValueError as e:
            reply = get_error_correction_prompt(str(e), max_length=self._max_error_length)
            response.message_type = MessageType.AI_RESPONSE_ERROR
            return response, reply

        # If the LLM doesn't generate a spec because the request is irrelevant,
        # return a dummy spec instead to avoid going into an error retrying loop.
        if (
            not self.config.retry_on_irrelevant_request
            and extracted_response.relevant_request == False
            and extracted_response.data_exists == False
            and spec_is_empty(extracted_spec)
        ):
            text_content = append_reply(
                "Invalid.",
                append_reply(extracted_response.relevant_request_rationale, extracted_response.data_exists_rationale),
            )
            assert text_content is not None
            response.spec = make_text_spec(text_content)
            response.model_response = extracted_response
            response.is_spec_fixed = False
            response.is_valid_schema = True
            response.is_empty_chart = True
            response.is_drawable = True
            return response, None
        else:
            spec_history = [v for (k, v) in sorted(self._spec_history.items())]
            spec_fix = validate_and_fix_spec(
                extracted_spec,
                self._df,
                retry_on_empty_plot=self.config.retry_on_empty_plot,
                max_reply_length=self._max_error_length,
                spec_history=spec_history,
            )
            assert spec_fix.spec_validity is not None

            response.spec = spec_fix.spec
            response.model_response = extracted_response
            response.is_spec_fixed = extracted_spec != spec_fix.spec
            response.is_valid_schema = spec_fix.spec_validity.is_valid_schema
            response.is_empty_chart = spec_fix.spec_validity.is_empty_scenegraph
            response.is_drawable = spec_fix.spec_validity.is_valid_scenegraph

            if not response.is_drawable or (self.config.retry_on_empty_plot and response.is_empty_chart):
                assert spec_fix.reply is not None
                response.message_type = MessageType.AI_RESPONSE_ERROR

            if spec_fix.reply is not None:
                # Not necessarily undrawable, but it causes a retry
                response.message_type = MessageType.AI_RESPONSE_ERROR
            return response, spec_fix.reply

    async def query(self, q: str, force_q: bool = False, message_type: MessageType = MessageType.USER) -> MessageInfo:
        # TODO just retry from start (different seed, temperature, ...)
        self._is_running = True
        content: str | None = q if force_q else self.get_user_prompt(q)
        n_attempts = 0
        while n_attempts <= self._n_retries and content is not None:
            msg_type = MessageType.USER_ERROR_CORRECTION if n_attempts > 0 else message_type
            response = await self._session.invoke(content, msg_type)
            response, content = self.process_response(response)
            if (spec_info := response.get_spec_info()) is not None:
                self._spec_history[len(self._session.messages) - 1] = spec_info
            n_attempts += 1
        self._is_running = False
        return self._session.last_message

    def select_chart(self, spec_info: SpecInfo) -> None:
        spec_info_prompt = get_select_spec_info_prompt(
            spec_info.spec, is_drawable=spec_info.is_drawable, is_empty_chart=spec_info.is_empty_chart
        )
        self.add_user_message(spec_info_prompt, MessageType.USER_ERROR_CORRECTION)
        self._spec_history[len(self._session.messages) - 1] = spec_info

    def clear_messages_from_index(self, message_index: int) -> None:
        """Revert the chat state to just before the given message index."""
        self.session.clear_messages_from_index(message_index)
        spec_history_indices = list(self._spec_history.keys())
        for i in spec_history_indices:
            if i >= message_index:
                del self._spec_history[i]

    @classmethod
    def from_config(cls, config: ModelConfig, df: pd.DataFrame, metadata: str = "", api_key: str | None = None) -> Self:
        chat_model = get_chat_model_from_config(config, api_key=api_key)
        return cls(
            config,
            df,
            chat_model,
            metadata=metadata,
        )


def get_chat_model_from_config(config: ModelConfig, api_key: str | None = None) -> BaseChatModel:
    secret_api_key = SecretStr(api_key) if api_key else None
    return ChatOpenAI(model=config.model_name, temperature=config.temperature, api_key=secret_api_key)
