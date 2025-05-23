import asyncio
import json
import pprint
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Literal, Self

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.load import Serializable
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from edaplot.data_prompts import DEFAULT_DATA_STRATEGY, DataDescriptionStrategy, get_data_description_prompt
from edaplot.prompt_utils import PromptTag, extract_json_tags
from edaplot.spec_utils import SpecType
from edaplot.vega import MessageType

logger = getLogger(__name__)


class RequestTypeTag(PromptTag):
    DATASET_RELEVANCE = "dataset_relevance"
    REQUEST_TYPE = "request_type"


@dataclass(kw_only=True)
class RequestTypeItem:
    type: str
    rationale: str
    answer: str


@dataclass(kw_only=True)
class RequestTypeMessage:
    message: BaseMessage
    message_type: MessageType
    response: dict[RequestTypeTag, RequestTypeItem] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        d["message_type"] = MessageType(d["message_type"])
        if d["response"] is not None:
            d["response"] = {
                RequestTypeTag(k): RequestTypeItem(**v) for k, v in d["response"].items() if k in RequestTypeTag
            }
        return cls(**d)


class MappedField(BaseModel):
    requested: str
    mapped_to: str


class DerivableField(BaseModel):
    requested: str
    derived_from: list[str]


class MissingField(BaseModel):
    requested: str
    fallback: str
    similarity: Literal["none", "low", "medium", "high"]


class DataAvailabilityOutput(Serializable, BaseModel):
    exact_fields: list[str]
    mapped_fields: list[MappedField]
    derivable_fields: list[DerivableField]
    missing_fields: list[MissingField]
    no_fields: bool
    missing_fields_explanation: str | None

    def to_json(self) -> dict[str, Any]:  # type: ignore
        # A hack for langchain serialization (TODO don't use langchain's serialization):
        # langchain dumps doesn't allow "default" kwarg (we need this to serialize pydantic models)
        # We also can't override default with a custom 'cls' because langchain uses a custom defaults
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        missing_fields = d.pop("missing_fields")
        if len(missing_fields) > 0 and isinstance(missing_fields[0], str):  # old format for loading benchmarks
            missing_fields = [MissingField(requested=f, fallback=f, similarity="none") for f in missing_fields]
        return cls(**d, missing_fields=missing_fields)


@dataclass(kw_only=True)
class DataAvailabilityMessage:
    message: BaseMessage
    message_type: MessageType
    response: DataAvailabilityOutput | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        d["message_type"] = MessageType(d["message_type"])
        if d["response"] is not None:
            d["response"] = DataAvailabilityOutput.from_dict(d["response"])
        return cls(**d)


@dataclass(kw_only=True)
class RequestAnalyzerOutput:
    request_type: list[RequestTypeMessage] | None = None
    data_availability: list[DataAvailabilityMessage] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        if d.get("request_type") is not None:
            d["request_type"] = [RequestTypeMessage.from_dict(m) for m in d["request_type"]]
        if d.get("data_availability") is not None:
            d["data_availability"] = [DataAvailabilityMessage.from_dict(m) for m in d["data_availability"]]
        return cls(**d)


@dataclass(kw_only=True)
class RequestAnalyzerConfig:
    model_name: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.0
    message_trimmer_max_tokens: int = 8192
    description_strategy: DataDescriptionStrategy = "main"

    analyze_data_availability: bool = True
    analyze_request_type: bool = False


class RequestAnalyzer:
    def __init__(
        self,
        config: RequestAnalyzerConfig,
        df: pd.DataFrame,
        model: BaseChatModel,
    ) -> None:
        self.config = config
        self.df = df
        self.model = model

        # Store the history as a map because the request analyzer can be disabled and enabled during the conversation
        self.response_history: dict[int, RequestAnalyzerOutput] = {}
        self._user_history: dict[int, str] = {}

    async def _analyze_data_availability(
        self, user_history: list[str], last_spec: SpecType | None = None
    ) -> list[DataAvailabilityMessage]:
        prompt = get_data_availability_prompt(
            self.df,
            user_history,
            data_description_strategy=self.config.description_strategy,
            last_spec=last_spec,
        )
        input_message = MessageType.create_message(prompt, MessageType.SYSTEM)
        messages = [
            DataAvailabilityMessage(message=input_message, message_type=MessageType.SYSTEM, response=None),
        ]

        # Using structured outputs
        structured_model = self.model.with_structured_output(DataAvailabilityOutput, include_raw=True)
        response_msg = await structured_model.ainvoke([m.message for m in messages])
        logger.info(response_msg)

        assert isinstance(response_msg, dict)
        ai_message = response_msg["raw"]
        ai_message.additional_kwargs.pop("parsed")  # no need to keep this twice
        response_message = DataAvailabilityMessage(
            message=ai_message, message_type=MessageType.AI_RESPONSE_VALID, response=response_msg["parsed"]
        )
        messages.append(response_message)
        return messages

    async def _analyze_request_type(
        self, user_history: list[str], last_spec: SpecType | None = None
    ) -> list[RequestTypeMessage]:
        prompt = get_request_type_prompt(
            self.df,
            user_history,
            data_description_strategy=self.config.description_strategy,
            last_spec=last_spec,
        )

        input_message = MessageType.create_message(prompt, MessageType.SYSTEM)
        messages = [
            RequestTypeMessage(message=input_message, message_type=MessageType.SYSTEM, response=None),
        ]
        response_msg = await self.model.ainvoke([m.message for m in messages])
        logger.info(response_msg)

        assert isinstance(response_msg.content, str)
        parsed_response = extract_request_type_response(response_msg.content)
        response_message = RequestTypeMessage(
            message=response_msg, message_type=MessageType.AI_RESPONSE_VALID, response=parsed_response
        )
        messages.append(response_message)
        return messages

    async def analyze_request(
        self, utterance: str, *, last_spec: SpecType | None = None, history_idx: int | None = None
    ) -> RequestAnalyzerOutput:
        if history_idx is None:
            history_idx = len(self.response_history)
        self._user_history[history_idx] = utterance

        user_history = [self._user_history[i] for i in sorted(self._user_history.keys())]
        async with asyncio.TaskGroup() as tg:
            if self.config.analyze_data_availability:
                data_availability_task = tg.create_task(self._analyze_data_availability(user_history, last_spec))
            if self.config.analyze_request_type:
                request_type_task = tg.create_task(self._analyze_request_type(user_history, last_spec))

        request_type_messages = None
        data_availability_messages = None
        if self.config.analyze_data_availability:
            data_availability_messages = data_availability_task.result()
        if self.config.analyze_request_type:
            request_type_messages = request_type_task.result()

        output = RequestAnalyzerOutput(
            request_type=request_type_messages,
            data_availability=data_availability_messages,
        )
        self.response_history[history_idx] = output
        return output

    def clear_messages_from_index(self, history_idx: int) -> None:
        history_indices = list(self.response_history.keys())
        for i in history_indices:
            if i >= history_idx:
                del self.response_history[i]
                del self._user_history[i]

    def get_response_history(self, history_idx: int) -> RequestAnalyzerOutput | None:
        return self.response_history.get(history_idx)

    def get_response_history_full(self) -> list[RequestAnalyzerOutput]:
        sorted_indices = sorted(self.response_history.keys())
        return [self.response_history[i] for i in sorted_indices]

    @classmethod
    def from_config(cls, config: RequestAnalyzerConfig, df: pd.DataFrame, api_key: str | None = None) -> Self:
        chat_model = get_chat_model_from_config(config, api_key=api_key)
        return cls(
            config,
            df,
            chat_model,
        )


def get_chat_model_from_config(config: RequestAnalyzerConfig, api_key: str | None = None) -> BaseChatModel:
    secret_api_key = SecretStr(api_key) if api_key else None
    return ChatOpenAI(model=config.model_name, temperature=config.temperature, api_key=secret_api_key)


def get_data_availability_prompt(
    df: pd.DataFrame,
    user_history: list[str],
    *,
    data_description_strategy: DataDescriptionStrategy = "main",
    last_spec: SpecType | None = None,
) -> str:
    data_description = get_data_description_prompt(df, data_description_strategy)
    vega_lite_json = json.dumps(last_spec, indent=2) if last_spec else None

    prompt = f"""\
You are a data analysis assistant evaluating if the requested data fields are available in the dataset. Please analyze the request considering the following:

1. Identify all data fields referenced in the request, including:
  - Exact matches: Fields that exactly match column names in the dataset
  - Semantic matches: Fields that refer to columns using different terms (e.g., "category" for "genre", "time" and "date", "region" and "country", "movie" and "title", etc.)
    - Implied fields (e.g., "over time" implies a temporal column, "in Japan" implies a country column)
    - Fields that can be inferred from the values of a column (e.g., "in USA" can be mapped to "Origin" if "Origin" contains locations such as "Canada", "Germany", etc.)
        - If the user is referencing specific values, assume that the values are available in the dataset (e.g. "United Kingdom" might be in a column "location" that contains country codes)
    - Pick the closest match to the user's request even if the fields don't have the same meaning (e.g., "province" and "state")
  - Derivable fields: Data that can be computed from existing columns:
    - Aggregations (mean, sum, count, min, max, etc.): "average sales" can be derived from "Sales"
    - Transformations (percentages, ratios, normalized values, etc.): "normalize counts" can be derived by aggregating to get a count and then normalizing
    - Grouping, filtering, formulas, etc...: "show rows where price > 100" can be derived from "price"; "month" or "year" can be derived from "date", etc.
  - Missing fields: Fields that are not available in the dataset and not derivable from existing columns
    - Do NOT include fields that fall into the other categories. This is a last resort for fields you are certain are missing.
        - If you categorized "price" into another category, don't mark "stock price" as missing because there is a large overlap.
    - If you mark a field as missing, you must also include the closest possible fallback column that the user could have meant instead, as well as the similarity between the requested term and the fallback column. If there is no suitable fallback, the similarity must be "none".

Important guidelines:
- Assume that the user is familiar with the dataset but might not be familiar with the exact column names.
- Focus on identifying truly missing data that would make the visualization impossible, while being flexible about field names and derivable values.
- Keep in mind that the user can make modification requests for a visualization or they can ask a different request from the previous one. 
- For modification and non-specific requests (e.g., "make it bigger", "change colors"), assume they reference the existing visualization and set "no_fields" to true.
- New requests should be evaluated independently of previous visualizations

2. Output your analysis in this JSON format:
```
{{
    "exact_fields": [list of requested fields that exist in the dataset],
    "mapped_fields": [{{"requested": "user_term", "mapped_to": "actual_column"}} for semantic matches],
    "derivable_fields": [{{"requested": "user_term", "derived_from": ["column1", "column2"]}} for derivable fields],
    "missing_fields": [{{"requested" "user_term", "fallback": "fallback column", "similarity": one of "none", "low", "medium", "high"}} for missing requested terms],
    "no_fields": true/false (for generic visualization/modification requests that don't reference any fields),
    "missing_fields_explanation": "One sentence that can be used to inform the user of missing fields. Do not mention the fallback column if the similarity is low."
}}
```

Example input: "show average sales per region, faceting by item type" and columns ["Sales", "Origin", "Genre", "Date"]
Example output:
```
{{
    "exact_fields": ["Sales"],
    "mapped_fields": [{{"requested": "region", "mapped_to": "Origin"}}],
    "derivable_fields": [{{"requested": "average sales", "derived_from": ["Sales"]}}],
    "missing_fields": [{{"requested": "item type", "fallback": "Genre", "similarity": "medium"}}],
    "no_fields": false,
    "missing_fields_explanation": "explain that 'item type' is missing"
}}
```

---

Dataset summary:
<dataset>
{data_description}
</dataset>

{f"Most recent Vega-Lite visualization:\n<vega_lite>\n{vega_lite_json}\n</vega_lite>" if vega_lite_json else ""}

Previous conversation context:
<user_history>
{user_history[:-1]}
</user_history>

Current request to analyze:
<current_request>
{user_history[-1]}
</current_request>
"""
    return prompt


def get_request_type_prompt(
    df: pd.DataFrame,
    user_history: list[str],
    *,
    data_description_strategy: DataDescriptionStrategy = DEFAULT_DATA_STRATEGY,
    last_spec: SpecType | None = None,
) -> str:
    data_description = get_data_description_prompt(df, data_description_strategy)
    vega_lite_json = json.dumps(last_spec, indent=2) if last_spec else None

    prompt = f"""\
You are a data analysis assistant evaluating user requests for data analysis and visualization. Your task is to determine if a request is valid and feasible given the available dataset.

Please analyze the user's request based on these two criteria:

1. Request Type ({RequestTypeTag.REQUEST_TYPE}):
  - What type of operation is being requested?
  - Example "visualization": "plot sales over time", "add legend", "change colors", ...
  - Example "transformation": "calculate average", "group by category", "normalize values", ...
  - Example "query": "show rows where price > 100", "filter sunny days", ...
  - Example "unknown": "make it better" (too vague) OR "compare to industry standard" (without context), ...

2. Data Availability ({RequestTypeTag.DATASET_RELEVANCE}):
  - Are the required data columns or properties available?
  - Can the requested operation be performed with the available data?
  - Example "true":
    * "Plot revenue vs time" (both columns exist)
    * "calculate mean price" (numeric column exists)
    * "group by category" (column exists)
  - Example "false":
    * "show profit margin" (no profit data)
    * "group by weather" (no weather column)
  - Example "unknown": "show the trends" (columns not specified)

Common operations to consider:
- Visualization: plotting, styling, formatting
- Aggregations: mean, sum, count, min, max
- Grouping: group by, aggregate
- Filtering: where, subset, query
- Transformations: normalize, scale
- Calculations: ratios, percentages, custom formulas
- Modifications: colors, titles, axis modifications

Important guidelines:
- Focus on analyzing the last request in the conversation history using the previous conversation as context
- Evaluate if the request is related to the most recent visualization or is a new, independent request
- For modification requests (e.g., "make it bigger", "add a title"), assume they reference the existing visualization
- For new requests unrelated to the previous visualization, evaluate them independently
- Consider implicit references to previous plots but don't assume all requests modify the existing visualization
- Treat visualization styling commands as valid even if they don't directly reference data columns
- Consider all types of data operations
- Check for semantic matches in column names
- Evaluate if operations can be derived from available data

Respond in this exact JSON format:
```
{RequestTypeTag.DATASET_RELEVANCE.wrap('{"rationale": "Brief explanation", "answer": "true"/"false"/"unknown"}')}
{RequestTypeTag.REQUEST_TYPE.wrap('{"rationale": "Brief explanation", "answer": "visualization"/"transformation"/"query"/"unknown"}')}
```

Dataset information:
<dataset>
{data_description}
</dataset>

{f"Most recent Vega-Lite visualization:\n<vega_lite>\n{vega_lite_json}\n</vega_lite>" if vega_lite_json else ""}

Previous conversation context:
<user_history>
{user_history[:-1]}
</user_history>

Current request to analyze:
<current_request>
{user_history[-1]}
</current_request>
"""
    return prompt


def extract_request_type_response(content: str) -> dict[RequestTypeTag, RequestTypeItem]:
    items = {}
    for tag in RequestTypeTag:
        item_list = extract_json_tags(content, tag)
        if len(item_list) > 0:
            items[tag] = RequestTypeItem(**item_list[0], type=tag.value)
    return items


def get_request_type_missing_data_warning(response: dict[RequestTypeTag, RequestTypeItem]) -> tuple[bool, str]:
    ok_item = RequestTypeItem(type="ok", answer="true", rationale="")  # for convenience
    data_not_available = response.get(RequestTypeTag.DATASET_RELEVANCE, ok_item).answer in [
        "false"
    ]  # add "unknown" for more sensitivity
    if not data_not_available:
        return False, ""
    warning_msg = "We detected that your request might reference non-existent data.\n"
    warning_msg += f"_{response.get(RequestTypeTag.DATASET_RELEVANCE, ok_item).rationale}_\n"
    return True, warning_msg


def get_request_type_warning(response: dict[RequestTypeTag, RequestTypeItem]) -> tuple[bool, str]:
    ok_item = RequestTypeItem(type="ok", answer="true", rationale="")  # for convenience
    request_type_unknown = response.get(RequestTypeTag.REQUEST_TYPE, ok_item).answer in ["unknown"]
    data_not_available = response.get(RequestTypeTag.DATASET_RELEVANCE, ok_item).answer in [
        "false"
    ]  # add "unknown" for more sensitivity
    warning_needed = request_type_unknown or data_not_available
    if not warning_needed:
        return False, ""

    warning_msg = "We detected that your request may be unclear or not feasible.\n"
    if request_type_unknown:
        warning_msg += (
            f"- The request type is unclear. _{response.get(RequestTypeTag.REQUEST_TYPE, ok_item).rationale}_\n"
        )
    if data_not_available:
        warning_msg += f"- The input dataset might not contain the required data. _{response.get(RequestTypeTag.DATASET_RELEVANCE, ok_item).rationale}_\n"
    return True, warning_msg


def get_data_availability_warning(output: DataAvailabilityOutput) -> tuple[bool, str]:
    if len(output.missing_fields) == 0 or output.no_fields:
        return False, ""

    # With this we can set the sensitivity of detection
    should_warn = any(missing_field.similarity in ["none", "low"] for missing_field in output.missing_fields)
    if not should_warn:
        return False, ""

    warning_msg = "We detected that your request might reference non-existent data.\n"
    warning_msg += f"_{output.missing_fields_explanation}_\n"
    return True, warning_msg


def get_request_analyzer_warning(output: RequestAnalyzerOutput) -> tuple[bool, str]:
    if output.data_availability is not None:
        data_availability_msg = output.data_availability[-1]
        if data_availability_msg.response is not None:
            return get_data_availability_warning(data_availability_msg.response)
    if output.request_type is not None:
        request_type_msg = output.request_type[-1]
        if request_type_msg.response is not None:
            return get_request_type_warning(request_type_msg.response)
    return False, ""


if __name__ == "__main__":

    def _interactive_cli() -> None:
        import asyncio

        import vega_datasets

        df = vega_datasets.data.cars()
        config = RequestAnalyzerConfig()
        model = RequestAnalyzer.from_config(config, df)
        while True:
            utterance = input("Enter a request: ")
            output = asyncio.run(model.analyze_request(utterance))
            pprint.pprint(output)

    _interactive_cli()
