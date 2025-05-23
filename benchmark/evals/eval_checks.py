import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from benchmark.evals.eval_types import ActionOutput, ActionType, InputAction
from benchmark.metrics import MetricInput, compute_metrics, jaccard_similarity
from benchmark.models.base_models import EvalMessage, EvalModelType
from edaplot.request_analyzer.header_analyzer import HeaderQuality
from edaplot.request_analyzer.request_analyzer import RequestAnalyzerOutput, get_request_analyzer_warning
from edaplot.spec_utils import SpecType, get_spec_color_ranges, get_spec_field, get_spec_field_by_path, get_spec_marks
from edaplot.transform_utils import get_transform_type, is_valid_transform


class CheckFunc(ABC):
    @abstractmethod
    def __call__(self, df: pd.DataFrame, actions: list[InputAction], outputs: list[ActionOutput]) -> dict[str, float]:
        pass


class CheckMark(CheckFunc):
    """Check if (any) output spec contains any of the specified mark types."""

    def __init__(self, mark_type: list[str]) -> None:
        self.ref_marks = set(mark_type)

    def __call__(self, df: pd.DataFrame, actions: list[InputAction], outputs: list[ActionOutput]) -> dict[str, float]:
        final_spec = _get_final_spec(outputs)
        if len(self.ref_marks) == 0 or final_spec is None:
            return {}
        score = 0.0
        hyp_marks = set(get_spec_marks(final_spec)) if final_spec is not None else set()
        if len(hyp_marks.intersection(self.ref_marks)) > 0:
            score = 1.0
        return {"check_mark": score}


class CheckGroundTruth(CheckFunc):
    def __init__(self, specs: list[SpecType]) -> None:
        self.ref_specs = specs

    def __call__(self, df: pd.DataFrame, actions: list[InputAction], outputs: list[ActionOutput]) -> dict[str, float]:
        if len(self.ref_specs) == 0:
            return {}

        check_prompts = [
            a.action_kwargs["user_utterance"] for a in actions if a.action_type == ActionType.USER_UTTERANCE
        ]
        chat_messages = [
            m.to_vega_message() for out in outputs if out.vega_chat_messages is not None for m in out.vega_chat_messages
        ]
        eval_messages = [
            EvalMessage(
                model_type=EvalModelType.VEGA_CHAT,
                message=m.message,
                message_type=m.message_type,
                spec_infos=m.spec_infos,
                explanation=m.explanation,
            )
            for m in chat_messages
        ]
        spec_score = 0.0
        # Return only max by spec score
        for ref_spec in self.ref_specs:
            metric_input = MetricInput(prompts=check_prompts, messages=eval_messages, ground_truth=ref_spec)
            ref_metrics = compute_metrics([metric_input])
            if ref_metrics["spec_score"] > spec_score:
                spec_score = ref_metrics["spec_score"]
        return {"spec_score": spec_score}


class CheckColors(CheckFunc):
    def __init__(self, colors: list[str]) -> None:
        self.ref_colors = colors

    @staticmethod
    def check_color_domains_valid(spec: SpecType, df: pd.DataFrame) -> bool:
        for color_channel in get_spec_field(spec, "color"):
            if "scale" not in color_channel:
                continue
            color_scale = color_channel["scale"]
            if not isinstance(color_scale, dict):
                continue
            if "domain" not in color_scale:
                continue
            color_domain = color_scale["domain"]
            if not isinstance(color_domain, list):
                continue
            possible_domain_values = None
            if "field" in color_channel:
                color_field = color_channel["field"]
                if isinstance(color_field, str) and color_field in df.columns:
                    possible_domain_values = df[color_field].unique()
            valid_domain_values = False
            if possible_domain_values is not None:
                valid_domain_values = set(color_domain).issubset(set(possible_domain_values))
            if not valid_domain_values:
                return False
        return True

    def __call__(self, df: pd.DataFrame, actions: list[InputAction], outputs: list[ActionOutput]) -> dict[str, float]:
        final_spec = _get_final_spec(outputs)
        if len(self.ref_colors) == 0 or final_spec is None:
            return {}
        sim = 0.0
        spec = copy.deepcopy(final_spec)
        for colors in get_spec_color_ranges(spec):
            s = jaccard_similarity(set(self.ref_colors), set(colors))
            sim = max(sim, s)
        if not self.check_color_domains_valid(spec, df):
            sim *= 0.5
        return {"check_colors": sim}


@dataclass
class CheckHasFieldKwargs:
    path: str
    value: Any


class CheckHasField(CheckFunc):
    def __init__(self, fields: list[CheckHasFieldKwargs]) -> None:
        self.ref_fields = fields

    def __call__(self, df: pd.DataFrame, actions: list[InputAction], outputs: list[ActionOutput]) -> dict[str, float]:
        final_spec = _get_final_spec(outputs)
        if len(self.ref_fields) == 0 or final_spec is None:
            return {}
        matched = 0.0
        for field_kwargs in self.ref_fields:
            path = tuple(field_kwargs.path.split("/"))
            hyp_fields = get_spec_field_by_path(final_spec, path)
            if len(hyp_fields) > 0:
                matched += max(1.0 if hyp_field == field_kwargs.value else 0.0 for _, hyp_field in hyp_fields)
        return {"check_has_fields": matched / len(self.ref_fields)}


class CheckTransform(CheckFunc):
    def __init__(self, transforms: list[SpecType], ignore_order: bool = True) -> None:
        self.ref_transforms = transforms
        self.ignore_order = ignore_order

    @staticmethod
    def compute_expression_similarity(expr_1: str, expr_2: str) -> float:
        # https://vega.github.io/vega-lite/docs/types.html#expression
        # We don't care about order, e.g.: `datum.temp * 2` = `2 * datum.temp`
        s1 = set(expr_1.split())
        s2 = set(expr_2.split())
        return jaccard_similarity(s1, s2)

    @staticmethod
    def filter_normalize_datetime(filt_dict: Any) -> Any:
        if isinstance(filt_dict, list):
            return [CheckTransform.filter_normalize_datetime(v) for v in filt_dict]
        elif isinstance(filt_dict, dict):
            datetime_keys = {"year", "month", "date", "hours", "minutes", "seconds", "milliseconds"}
            filt_keys = set(filt_dict.keys())
            if filt_keys.issubset(datetime_keys):
                return {
                    "datetime": pd.Timestamp(
                        year=filt_dict.get("year", 0),
                        month=filt_dict.get("month", 1),
                        day=filt_dict.get("date", 1),
                        hour=filt_dict.get("hours"),
                        minute=filt_dict.get("minutes"),
                        second=filt_dict.get("seconds"),
                        microsecond=filt_dict.get("milliseconds", 0) * 1000,
                    )
                }
            new_filt = {}
            for k, v in filt_dict.items():
                new_filt[k] = CheckTransform.filter_normalize_datetime(v)
            return new_filt
        else:
            return filt_dict

    def __call__(self, df: pd.DataFrame, actions: list[InputAction], outputs: list[ActionOutput]) -> dict[str, float]:
        hyp_spec = _get_final_spec(outputs)
        if len(self.ref_transforms) == 0 or hyp_spec is None:
            return {}
        hyp_transforms = hyp_spec.get("transform", [])

        hyp_type_to_idx = {}
        for i, hyp_transform in enumerate(hyp_transforms):
            hyp_type_to_idx[get_transform_type(hyp_transform)] = i

        matched = 0.0
        for ref_transform in self.ref_transforms:
            ref_type = get_transform_type(ref_transform)
            if ref_type not in hyp_type_to_idx:
                continue
            hyp_transform = hyp_transforms[hyp_type_to_idx[ref_type]]
            if not is_valid_transform(hyp_spec, hyp_transform):
                continue
            match = 0.0
            if ref_type == "filter":
                ref_filter = self.filter_normalize_datetime(ref_transform)
                hyp_filter = self.filter_normalize_datetime(hyp_transform)
                match = 1.0 if ref_filter == hyp_filter else 0.0
            elif ref_type == "calculate":
                match = self.compute_expression_similarity(ref_transform["calculate"], hyp_transform["calculate"])
            matched += match
        return {"check_transform": matched / len(self.ref_transforms)}


class CheckRequestAnalyzer(CheckFunc):
    def __init__(self, should_warn: bool) -> None:
        self.should_warn = should_warn

    @staticmethod
    def get_request_analyzer_response(
        outputs: list[ActionOutput],
    ) -> RequestAnalyzerOutput | None:
        if len(outputs) == 0 or outputs[-1].request_analyzer_history is None:
            return None
        final_response = outputs[-1].request_analyzer_history[-1]
        if final_response is None:
            return None
        return final_response

    def __call__(self, df: pd.DataFrame, actions: list[InputAction], outputs: list[ActionOutput]) -> dict[str, float]:
        response = self.get_request_analyzer_response(outputs)
        if response is None:
            return {}
        warning_needed, _ = get_request_analyzer_warning(response)
        score = 1.0 if self.should_warn == warning_needed else 0.0
        return {"check_request_analyzer": score}


class CheckHeaderAnalyzer(CheckFunc):
    def __init__(self, unclear: list[str]) -> None:
        self.unclear = unclear

    def __call__(self, df: pd.DataFrame, actions: list[InputAction], outputs: list[ActionOutput]) -> dict[str, float]:
        if len(outputs) == 0 or outputs[-1].header_analyzer_history is None:
            return {}
        final_response = outputs[-1].header_analyzer_history[-1]
        if final_response is None or final_response.response is None:
            return {}
        unclear_columns = final_response.response.get(HeaderQuality.UNCLEAR, [])
        score = jaccard_similarity(set(self.unclear), set(c.column_name for c in unclear_columns))
        return {"check_header_analyzer": score}


def _get_final_spec(outputs: list[ActionOutput]) -> SpecType | None:
    if len(outputs) == 0 or outputs[-1].vega_chat_messages is None:
        return None
    final_message = outputs[-1].vega_chat_messages[-1]
    return final_message.spec
