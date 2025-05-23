import dataclasses
import itertools
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, TypeAlias

import langchain_core.load
import pandas as pd
import yaml

from benchmark.datasets import load_dataset_from_config
from edaplot.request_analyzer.header_analyzer import HeaderAnalyzerMessage
from edaplot.request_analyzer.request_analyzer import RequestAnalyzerOutput, RequestTypeMessage
from edaplot.vega import VegaMessage
from edaplot.vega_chat.vega_chat import MessageInfo


class CheckType(StrEnum):
    MARK = "mark"
    GROUND_TRUTH = "ground_truth"
    COLORS = "colors"
    HAS_FIELD = "has_field"
    TRANSFORM = "transform"
    REQUEST_ANALYZER = "request_analyzer"
    HEADER_ANALYZER = "header_analyzer"


class ActionType(StrEnum):
    USER_UTTERANCE = "user_utterance"
    SELECT_CHART = "select_chart"
    HEADER_ANALYZER = "header_analyzer"


@dataclass(kw_only=True)
class TargetCheck:
    check_type: CheckType
    check_kwargs: dict[str, Any]


@dataclass(kw_only=True)
class InputAction:
    action_type: ActionType
    action_kwargs: dict[str, Any]
    checks: list[TargetCheck] = dataclasses.field(default_factory=list)


@dataclass(kw_only=True)
class InputData:
    dataset_config: dict[str, Any] | None = None
    dataset_item_id: str | None = None
    path: str | None = None

    @property
    def dataset_name(self) -> str | None:
        if self.path is not None:
            return Path(self.path).stem
        elif self.dataset_config is not None:
            return self.dataset_config["name"]
        return None


@dataclass(kw_only=True)
class EvalInput:
    id: str
    data: InputData
    actions: list[InputAction]

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        with open(path) as f:
            inp = yaml.safe_load(f)
        if inp["id"] is None:
            inp["id"] = path.stem
        return cls.from_dict(inp)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        actions = []
        for action in d["actions"]:
            action_type = ActionType(action["action_type"])
            action_kwargs = action["action_kwargs"]
            checks = [
                TargetCheck(check_type=CheckType(check.pop("check_type")), **check)
                for check in action.get("checks", [])
            ]
            actions.append(InputAction(action_type=action_type, action_kwargs=action_kwargs, checks=checks))
        return cls(id=d["id"], data=InputData(**d["data"]), actions=actions)

    def load_dataframe(self) -> pd.DataFrame:
        # We could have an action to set the dataset instead, but for now we always have it as a single required input
        if self.data.path is not None:
            return pd.read_csv(self.data.path)  # assume csv for now
        assert self.data.dataset_config is not None
        assert self.data.dataset_item_id is not None
        dataset = load_dataset_from_config(self.data.dataset_config)
        return dataset[self.data.dataset_item_id].data


@dataclass(kw_only=True)
class ActionOutput:
    vega_chat_messages: list[MessageInfo] | None = None  # All vega chat messages during this action
    request_analyzer_history: list[RequestAnalyzerOutput | None] | None = (
        None  # All request analyzer messages for each chat message during this action
    )
    header_analyzer_history: list[HeaderAnalyzerMessage] | None = (
        None  # All header analyzer messages during this action
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        vega_chat_messages = d.pop("vega_chat_messages", None)
        if vega_chat_messages is not None:
            vega_chat_messages = [MessageInfo.from_dict(m) for m in vega_chat_messages]
        request_analyzer_history = d.pop("request_analyzer_history", None)
        if request_analyzer_history is not None:
            _request_analyzer_history: list[RequestAnalyzerOutput | None] = []
            for m in request_analyzer_history:
                if m is None:
                    _request_analyzer_history.append(None)
                elif "history_idx" in m and "messages" in m:  # old format
                    _request_analyzer_history.append(
                        RequestAnalyzerOutput(
                            request_type=[RequestTypeMessage.from_dict(msg) for msg in m["messages"]],
                        )
                    )
                else:
                    _request_analyzer_history.append(RequestAnalyzerOutput.from_dict(m))
            request_analyzer_history = _request_analyzer_history
        header_analyzer_history = d.pop("header_analyzer_history", None)
        if header_analyzer_history is not None:
            header_analyzer_history = [HeaderAnalyzerMessage.from_dict(m) for m in header_analyzer_history]
        return cls(
            vega_chat_messages=vega_chat_messages,
            request_analyzer_history=request_analyzer_history,
            header_analyzer_history=header_analyzer_history,
        )


EvalCheckResult: TypeAlias = dict[CheckType, dict[str, float]]


@dataclass(kw_only=True)
class EvalOutput:
    input: EvalInput  # Copy of the input for convenience and reference
    action_outputs: list[ActionOutput]
    check_results: list[EvalCheckResult]

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        return langchain_core.load.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> Self:
        # Use langchain's json serialization to deal with BaseMessage
        d = langchain_core.load.loads(s)

        # Backwards compatibility with old eval format
        if "messages" in d and "check_indices" in d:
            messages = []
            for dict_msg in d.pop("messages"):
                vega_msg = VegaMessage.from_dict(dict_msg)
                spec_info = vega_msg.spec_infos[0] if len(vega_msg.spec_infos) > 0 else None
                messages.append(
                    MessageInfo(
                        message=vega_msg.message,
                        message_type=vega_msg.message_type,
                        spec=spec_info.spec if spec_info is not None else None,
                        is_spec_fixed=spec_info.is_spec_fixed if spec_info is not None else False,
                        is_empty_chart=spec_info.is_empty_chart if spec_info is not None else True,
                        is_valid_schema=spec_info.is_valid_schema if spec_info is not None else False,
                        is_drawable=spec_info.is_drawable if spec_info is not None else False,
                        model_response=None,
                    )
                )

            action_outputs = []
            for check_i, check_j in itertools.pairwise([0] + d.pop("check_indices")):
                action_outputs.append(ActionOutput(vega_chat_messages=messages[check_i + 1 : check_j + 1]))
        else:
            action_outputs = [ActionOutput.from_dict(m) for m in d.pop("action_outputs")]

        d["input"] = EvalInput.from_dict(d["input"])
        return cls(**d, action_outputs=action_outputs)
