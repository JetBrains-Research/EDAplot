import argparse
import logging
import pprint
import typing
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy  # type: ignore
from langchain_core.messages import AIMessage, BaseMessage

from benchmark.lida_self_eval import LIDASelfEvalScore
from benchmark.models.base_models import EvalMessage
from edaplot.spec_utils import (
    get_spec_field,
    get_spec_keys,
    get_spec_leaf_key_values,
    get_spec_marks,
    get_spec_paths,
    get_spec_transform_paths,
    spec_paths_ignore_list_order,
)
from edaplot.vega import MessageType, SpecInfo, SpecType

logger = logging.getLogger(__name__)

# See: https://vega.github.io/vega-lite/docs/mark.html
VEGA_LITE_MARKS = [
    "bar",
    "circle",
    "square",
    "tick",
    "line",
    "area",
    "point",
    "geoshape",
    "rule",
    "text",
    "boxplot",
    "errorband",
    "errorbar",
]
# The user may specify a mark with a synonym:
# E.g. "column chart" == "bar chart"
VEGA_LITE_MARKS_SYNONYMS = [
    "donut",
    "map",
    "pie",
    "heatmap",
    "dot",
    "array",
    "scatterplot",
    "scatter",
    "histogram",
    "column",
    "row",
    "cluster",  # used for scatter plots...
]

# N.B. we use f1 even if beta != 1 (because f1 is cooler than f)
F1Score = typing.NamedTuple("F1Score", [("f1", float), ("precision", float), ("recall", float)])


@dataclass
class MetricInput:
    """Wrapper around saved messages and outputs for computing metrics."""

    prompts: list[str]
    messages: list[EvalMessage]
    ground_truth: SpecType | None = None  # e.g. token stats don't need this
    lida_self_eval_scores: list[LIDASelfEvalScore] | None = None
    vision_judge_score: float | None = None
    vision_judge_is_empty_chart: bool | None = None

    @property
    def final_message(self) -> EvalMessage:
        return self.messages[-1]

    @property
    def final_spec_info(self) -> SpecInfo | None:
        spec_infos = self.final_message.spec_infos
        if len(spec_infos) == 0:
            return None
        return spec_infos[0]


@dataclass(frozen=True)
class MetricValue:
    value: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    def to_dict(self, name: str) -> dict[str, float]:
        return {
            name: self.value,
            f"{name}_ci_low": self.ci_low,
            f"{name}_ci_high": self.ci_high,
        }


def sample_mean_confidence_interval(xs: list[float], conf: float = 0.95) -> MetricValue:
    """Compute a `t` confidence interval for the sample mean.

    Make sure the conditions are met:
    - Random: A random sample or randomized experiment should be used to obtain the data.
    - Normal: The sampling distribution of the sample mean needs to be approximately normal.
    This is true if our parent population is normal or if our sample is reasonably large (n >= 30).
    - Independent: Individual observations need to be independent. If sampling without replacement, our sample size shouldn't be more than 10% of the population.

    See https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample/estimating-population-mean/a/reference-conditions-inference-one-mean.
    """
    # From https://stackoverflow.com/a/34474255
    xs = [x for x in xs if not np.isnan(x)]
    n = len(xs)
    if n == 0:
        return MetricValue(float("nan"))
    elif n == 1:
        return MetricValue(xs[0])
    mean = np.mean(xs, dtype=float)
    ci_low, ci_high = scipy.stats.t.interval(confidence=conf, df=n - 1, loc=mean, scale=scipy.stats.sem(xs))
    return MetricValue(mean, ci_low, ci_high)


def bootstrap_confidence_interval(
    xs: Sequence[float],
    statistic_fn: typing.Callable[[np.ndarray], float],
    conf: float = 0.95,
) -> MetricValue:
    """Compute a bootstrapped confidence interval for a given statistic."""
    # When in doubt, bootstrap.
    xs_clean = [x for x in xs if not np.isnan(x)]
    n = len(xs_clean)
    if n == 0:
        return MetricValue(float("nan"))
    data_array = np.array(xs_clean)
    observed_statistic = statistic_fn(data_array)
    if n == 1:
        return MetricValue(observed_statistic)
    result = scipy.stats.bootstrap(
        data=(data_array,),
        statistic=statistic_fn,
        confidence_level=conf,
        random_state=42,
    )
    ci_low = result.confidence_interval.low
    ci_high = result.confidence_interval.high
    return MetricValue(observed_statistic, ci_low, ci_high)


def f_beta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    # https://en.wikipedia.org/wiki/F-score
    # > beta is chosen such that recall is considered beta times as important as precision
    if precision + recall > 0.0:
        b2 = beta * beta
        return (1.0 + b2) * precision * recall / ((b2 * precision) + recall)
    else:
        return 0.0


def compute_f1(list_ref: list, list_hyp: list, beta: float = 1.0) -> F1Score:
    """Compute F1 score from the reference (ground truth) list and hypothesis list.

    Args:
      list_ref: List of true elements.
      list_hyp: List of positive (retrieved) elements.

    Returns:
      A F1Score object containing F_beta, precision, and recall scores.
    """
    # Adapted from: https://github.com/google-research/google-research/blob/master/schema_guided_dst/metrics.py
    ref = Counter(list_ref)
    hyp = Counter(list_hyp)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    f = f_beta_score(precision, recall, beta=beta)
    return F1Score(f1=f, precision=precision, recall=recall)


def compute_f1_weighted(
    list_ref: list[tuple[Any, float]], list_hyp: list[tuple[Any, float]], beta: float = 1.0
) -> F1Score:
    """Compute sample-weighted F1 score from the reference (ground truth) list and hypothesis list.
    All weights must be positive.

    Args:
      list_ref: List of true elements and their weights.
      list_hyp: List of positive (retrieved) elements and their weights.

    Returns:
      A F1Score object containing F_beta, precision, and recall scores.
    """
    # Counter works with floats as well
    ref: Counter[float] = Counter()
    for item, weight in list_ref:
        ref[item] += weight  # type: ignore
    hyp: Counter[float] = Counter()
    for item, weight in list_hyp:
        hyp[item] += weight  # type: ignore
    true = sum(ref.values())  # sum of weights
    positive = sum(hyp.values())  # sum of weights
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    f = f_beta_score(precision, recall, beta=beta)
    return F1Score(f1=f, precision=precision, recall=recall)


def compute_f1_mean(f1_scores: list[F1Score]) -> F1Score:
    f1s = [f1.f1 for f1 in f1_scores if not np.isnan(f1.f1)]
    precisions = [f1.precision for f1 in f1_scores if not np.isnan(f1.precision)]
    recalls = [f1.recall for f1 in f1_scores if not np.isnan(f1.recall)]
    return F1Score(
        f1=np.mean(f1s, dtype=float) if len(f1s) > 0 else float("nan"),
        precision=np.mean(precisions, dtype=float) if len(precisions) > 0 else float("nan"),
        recall=np.mean(recalls, dtype=float) if len(recalls) > 0 else float("nan"),
    )


def get_messages_token_usage(messages: list[BaseMessage]) -> dict[str, float]:
    sum_input_tokens = 0
    sum_output_tokens = 0
    sum_total_tokens = 0
    for message in messages:
        if isinstance(message, AIMessage) and (usage := message.usage_metadata) is not None:
            sum_input_tokens += usage["input_tokens"]
            sum_output_tokens += usage["output_tokens"]
            sum_total_tokens += usage["total_tokens"]
    return {
        "input_tokens": sum_input_tokens,
        "output_tokens": sum_output_tokens,
        "total_tokens": sum_total_tokens,
    }


def aggregate_token_counts(
    counts: dict[str, list[float]], percentiles: Sequence[float] = (0.5, 0.95, 0.99)
) -> dict[str, float]:
    out = {}
    for k, num_tokens in counts.items():
        out[f"{k}_mean"] = np.mean(num_tokens, dtype=float) if len(num_tokens) > 0 else float("nan")
        for perc in percentiles:
            out[f"{k}_{perc:.2f}"] = np.percentile(num_tokens, perc * 100) if len(num_tokens) > 0 else float("nan")
    return out


def token_usage_stats(outputs: list[MetricInput], percentiles: Sequence[float] = (0.5, 0.95, 0.99)) -> dict[str, float]:
    counts: dict[str, list[float]] = {"input_tokens": [], "output_tokens": [], "total_tokens": []}
    for output in outputs:
        message_token_stats = get_messages_token_usage([m.message for m in output.messages])
        for k, n_tokens in message_token_stats.items():
            counts[k].append(n_tokens)
    return aggregate_token_counts(counts, percentiles)


def visualization_error_rate(outputs: list[MetricInput]) -> MetricValue:
    """Visualization error rate is computed as the percentage of generated visualizations
    that result in code compilation errors [defined in LIDA (4.1.1)].

    Also known as Syntax Correctness [vi(E)va].
    """
    errors = []
    for output in outputs:
        spec_info = output.final_spec_info
        if spec_info is not None:
            errors.append(0 if spec_info.is_drawable else 1)
        elif output.final_message.base64_raster is None:
            # For code-output models, we check if we rendered the chart
            errors.append(1)
        else:
            errors.append(0)
    return bootstrap_confidence_interval(errors, np.mean, conf=0.95)


def response_error_rate(outputs: list[MetricInput]) -> float:
    """Percentage of all model responses that contain an error (not just final responses,
    but intermediate responses as well)."""
    errors = 0
    total = 0
    for output in outputs:
        for msg in output.messages:
            if MessageType.is_ai_response(msg.message_type):
                total += 1
                if MessageType.is_ai_response_error(msg.message_type):
                    errors += 1
    return errors / total if total > 0 else float("nan")


def empty_plot_error_rate(outputs: list[MetricInput]) -> MetricValue:
    """Percentage of all final plots that are empty. An invalid plot is also defined to be empty."""
    errors = []
    for output in outputs:
        spec_info = output.final_spec_info
        if spec_info is not None:
            errors.append(1 if spec_info.is_empty_chart else 0)
        elif output.vision_judge_is_empty_chart is not None:
            # Use vision judge for non-VL outputs
            errors.append(1 if output.vision_judge_is_empty_chart else 0)
        else:
            errors.append(1)
    return bootstrap_confidence_interval(errors, np.mean, conf=0.95)
    # return sample_mean_confidence_interval(errors, conf=0.95)


def spec_f1_correctness_mark(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    """Correctness of mark types (e.g. 'bar' and 'line') between the two specs.
    Generalization of Mark Correctness from [vi(E)va].

    Matches examples such as: `{"mark": "bar"} === {"mark": {"type": "bar"}}`

    The following marks are counted as 1/2 equivalent: `circle` == `point` == `square`.
    """

    def apply_mark_equivalence(marks: list[str]) -> list[str]:
        out = []
        for mark in marks:
            if mark in ("circle", "point", "square"):
                out.append("circle-point-square")
            else:
                out.append(mark)
        return out

    hyp_marks = get_spec_marks(spec_hyp)
    ref_marks = get_spec_marks(spec_ref)
    orig_f1 = compute_f1(ref_marks, hyp_marks)
    equiv_f1 = compute_f1(apply_mark_equivalence(ref_marks), apply_mark_equivalence(hyp_marks))
    # orig_f1 <= equiv_f1
    return F1Score(
        f1=(orig_f1.f1 + equiv_f1.f1) / 2,
        precision=(orig_f1.precision + equiv_f1.precision) / 2,
        recall=(orig_f1.recall + equiv_f1.recall) / 2,
    )


def get_my_encoding_fields(spec: SpecType, *, include_titles: bool = False) -> list[tuple[str, str, Any]]:
    # Check: https://vega.github.io/vega-lite/docs/encoding.html
    check_encoding_channels = (
        ["x", "y", "x2", "y2", "xError", "yError", "xError2", "yError2"]
        + ["color", "row", "column"]  # This is for creating groups and facets
        + ["theta", "theta2", "radius", "radius2"]
        + ["longitude", "latitude", "longitude2", "latitude2"]
    )
    check_field_properties = ["field", "type", "aggregate", "bin", "timeUnit"] + (
        ["title", "axis"] if include_titles else []
    )
    values = []
    # N.B. The generated spec may not be valid, so we can't make any required value/type assumptions
    # TODO similarity based `title` comparison
    for enc in get_spec_field(spec, "encoding"):
        if not isinstance(enc, dict):
            continue
        for ch in check_encoding_channels:
            if ch not in enc:
                continue
            for field in check_field_properties:
                if field not in enc[ch]:
                    continue
                value = enc[ch][field]
                if field == "field" and isinstance(enc[ch][field], dict):
                    if "repeat" in enc[ch][field]:
                        value = enc[ch][field]["repeat"]
                    else:
                        continue
                elif field == "bin" and isinstance(enc[ch][field], dict):
                    continue  # Check just for "bin": true/false # TODO act as if "true" in this case
                elif field == "title" and isinstance(enc[ch][field], list):
                    value = "".join(enc[ch][field])
                elif field == "axis" and isinstance(enc[ch][field], dict):
                    if "title" in enc[ch][field]:
                        # We want to match `encoding/x/title` with `encoding/x/axis/title`
                        value = enc[ch][field]["title"]
                        field = "title"
                    else:
                        continue
                values.append((ch, field, value))
    return values


def spec_f1_correctness_encoding(
    spec_ref: SpecType,
    spec_hyp: SpecType,
    *,
    swappable_xy: bool = True,
    swappable_faceting: bool = True,
    include_titles: bool = False,
    types_weight: float = 0.5,
    time_unit_weight: float = 0.5,
    beta: float = 2.0,
) -> F1Score:
    """Correctness of the mapping between the dataset columns and axes channels
    (e.g., `x/field/foo` and `x/type/nominal`).

    - Only data-specific encoding channels and fields are compared
      (e.g. `x`, `y`, `field`, `type`, `aggregate`, `bin`, ...).
    - The x and y axes can optionally be swapped, i.e. `x/field/foo == y/field/foo` (default: True, as in [NLV]).
    - Row and column faceting can be swappable, i.e. `encoding/row/field/foo == encoding/column/field/foo` (default: True).
    - `type` fields are weighted less (default: 0.5).
    - `timeUnit` fields are weighted less (default: 0.5).
    - Axis titles can optionally be compared (default: False).
    - Recall is valued more than precision.

    N.B. `transform` operations can be either in `encoding` (aggregate, bin, ...) or in the view-level `transform`.
    """

    def do_swappable_fields(values: list[tuple[str, str, Any]]) -> list[tuple[str, str, Any]]:
        out = []
        for ch, field, value in values:
            if swappable_xy and ch[0] in ("x", "y"):  # For x, y, x2, y2, ...
                # TODO penalty for xy swap
                ch = "xy" + ch[1:]
            elif swappable_faceting and ch in ("row", "column"):
                ch = "row|column"
            out.append((ch, field, value))
        return out

    def apply_weights(values: list[tuple[str, str, Any]]) -> list[tuple[tuple[str, str, Any], float]]:
        out = []
        for ch, field, value in values:
            w = 1.0
            if field == "type":
                w = types_weight
            elif field == "timeUnit":
                w = time_unit_weight
            out.append(((ch, field, value), w))
        return out

    hyp_values = get_my_encoding_fields(spec_hyp, include_titles=include_titles)
    ref_values = get_my_encoding_fields(spec_ref, include_titles=include_titles)
    hyp_values = apply_weights(do_swappable_fields(hyp_values))
    ref_values = apply_weights(do_swappable_fields(ref_values))
    return compute_f1_weighted(ref_values, hyp_values, beta=beta)


def spec_f1_correctness_transform(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    """Correctness of the **view-level** [transforms](https://vega.github.io/vega-lite/docs/transform.html)
    between the reference and hypothesis specs."""

    def get_normalized_paths(spec: SpecType) -> list[tuple]:
        paths = get_spec_transform_paths(spec)
        return spec_paths_ignore_list_order(paths)

    ref_values = get_normalized_paths(spec_ref)
    hyp_values = get_normalized_paths(spec_hyp)
    return compute_f1(ref_values, hyp_values)


def spec_f1_correctness_full(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    """Correctness of *all* fields between the reference and hypothesis specs.

    For example:
    ```
    {
      "mark": "line",
      "encoding": {
        "x": {
          "field": "date"
        }
      }
    }
    ```
    will compare `mark/line` and `encoding/x/field/date` to the other spec.
    """

    def get_normalized_paths(spec: SpecType) -> list[tuple]:
        paths = get_spec_paths(spec)
        return spec_paths_ignore_list_order(paths)

    hyp_paths = get_normalized_paths(spec_hyp)
    ref_paths = get_normalized_paths(spec_ref)
    return compute_f1(ref_paths, hyp_paths)


def spec_f1_correctness_key_values(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    """Correctness of all key-value pairs between the reference and hypothesis specs.

    For example:
    ```
    {
      "mark": "line",
      "encoding": {
        "x": {
          "field": "date"
        }
      }
    }
    ```
    will compare `mark/line` and `field/date` to the other spec.
    Notably, `field/date` will match even if the other spec is on the `y` axis.
    """
    hyp_kvs = get_spec_leaf_key_values(spec_hyp)
    ref_kvs = get_spec_leaf_key_values(spec_ref)
    return compute_f1(ref_kvs, hyp_kvs)


def spec_f1_correctness_key(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    """Correctness of all keys between the reference and hypothesis specs.

    For example:
    ```
    {
      "mark": "line",
      "encoding": {
        "x": {
          "field": "date"
        }
      }
    }
    ```
    will compare the presence of `mark`, `encoding`, `x` and `field` to the other spec.
    """
    ref_keys = get_spec_keys(spec_ref)
    hyp_keys = get_spec_keys(spec_hyp)
    return compute_f1(ref_keys, hyp_keys)


def jaccard_similarity(a: set[Any], b: set[Any]) -> float:
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else float("nan")


def spec_jaccard_keys(spec_ref: SpecType, spec_hyp: SpecType) -> float:
    """Jaccard Similarity between the sets of keys in the schemas of the two visualizations [vi(E)va].

    Also known as Grammar Similarity [vi(E)va].
    This metric doesn't take repeated keys into account
    (e.g., if 'field' occurs multiple times, it will be counted only once).
    """
    ref_keys = set(get_spec_keys(spec_ref))
    hyp_keys = set(get_spec_keys(spec_hyp))
    return jaccard_similarity(ref_keys, hyp_keys)


def get_marks_in_utterance(utterance: str) -> list[str]:
    # N.B. We don't know if the user is saying, "Make a scatter plot" or "DON'T make a scatter plot"
    # TODO extract with a (local?) LLM
    utt_marks = []
    mark_keywords = set(VEGA_LITE_MARKS + VEGA_LITE_MARKS_SYNONYMS)
    for w in utterance.lower().split():
        mark_candidate = w.strip()
        if mark_candidate in mark_keywords:
            utt_marks.append(mark_candidate)
    return utt_marks


def spec_score_impl(
    spec_ref: SpecType,
    spec_hyp: SpecType,
    *,
    utterance: str,
    hyp_is_drawable: bool,
    hyp_is_empty_chart: bool,
    hyp_is_valid_schema: bool,
) -> float:
    if not hyp_is_drawable:
        return 0.0

    # Relative weights/importance of each feature
    w_drawable = 0.005  # Give some score just for spec_hyp being drawable
    x_drawable = w_drawable
    w_valid_schema = w_drawable if hyp_is_valid_schema else 1.0
    x_valid_schema = w_valid_schema if hyp_is_valid_schema else 0.0

    # Give some score just for not making an empty plot, otherwise a huge penalty.
    # We don't return here because we want to slightly differentiate between an empty plot that is nearly correct
    # and an empty plot that is very wrong
    w_not_empty = 1000.0 if hyp_is_empty_chart else w_drawable
    x_not_empty = 0.0 if hyp_is_empty_chart else w_not_empty

    w_encoding_penalty = 0.0  # percentage-based

    ref_transforms = get_spec_transform_paths(spec_ref)
    if len(ref_transforms) > 0:
        w_transform = 1.0
        x_transform = w_transform * spec_f1_correctness_transform(spec_ref, spec_hyp).f1
    else:
        hyp_transforms = get_spec_transform_paths(spec_hyp)
        if len(hyp_transforms) > 0:
            # Penalize adding transforms when the ref can be made without transforms
            # (maybe the end result is equivalent, but that is rare)
            # Trust encodings less in this case (since transforms can add new 'as' fields)
            w_transform = 1.0
            x_transform = 0.0
            w_encoding_penalty += 0.25
        else:
            # Give some score for not including view-level transforms unnecessarily
            # (since we prefer inline transforms); f1 is 1.0 in this case
            w_transform = w_drawable
            x_transform = w_transform

    # Increase mark importance if some mark is specified in the utterance
    utt_marks = get_marks_in_utterance(utterance)
    w_mark = 1.0 if len(utt_marks) > 0 else 0.5
    x_mark = w_mark * spec_f1_correctness_mark(spec_ref, spec_hyp).f1

    w_encoding_multiplier = max(0.0, (1.0 - w_encoding_penalty))
    w_encoding = w_encoding_multiplier * 3.0
    x_encoding = w_encoding * spec_f1_correctness_encoding(spec_ref, spec_hyp).f1

    score = x_drawable + x_valid_schema + x_not_empty + x_encoding + x_mark + x_transform
    w_sum = w_drawable + w_valid_schema + w_not_empty + w_encoding + w_mark + w_transform
    return score / w_sum


def compute_spec_score(output: MetricInput) -> float:
    """Single score that measures how well the generated spec matches the ground truth spec.
    The score combines schema, mark, encoding, and transform correctness.
    Empty plots are heavily penalized.
    _Style_ comparison is currently ignored.
    """
    spec_ref = output.ground_truth
    if spec_ref is None:
        return float("nan")
    spec_info = output.final_spec_info
    if spec_info is None:
        return 0.0
    utterance = "\n".join(output.prompts)  # Include the whole history for multi-turn prompts
    return spec_score_impl(
        spec_ref,
        spec_info.spec,
        utterance=utterance,
        hyp_is_drawable=spec_info.is_drawable,
        hyp_is_empty_chart=spec_info.is_empty_chart,
        hyp_is_valid_schema=spec_info.is_valid_schema,
    )


def compute_spec_metrics(spec_ref: SpecType, spec_hyp: SpecType) -> dict[str, float]:
    f1_mark = spec_f1_correctness_mark(spec_ref, spec_hyp)
    f1_encoding = spec_f1_correctness_encoding(spec_ref, spec_hyp)
    f1_transform = spec_f1_correctness_transform(spec_ref, spec_hyp)
    f1_full = spec_f1_correctness_full(spec_ref, spec_hyp)
    f1_keys = spec_f1_correctness_key(spec_ref, spec_hyp)
    f1_kvs = spec_f1_correctness_key_values(spec_ref, spec_hyp)
    jaccard_keys = spec_jaccard_keys(spec_ref, spec_hyp)
    return {
        **{f"mark_{k}": v for k, v in f1_mark._asdict().items()},
        **{f"encoding_{k}": v for k, v in f1_encoding._asdict().items()},
        **{f"transform_{k}": v for k, v in f1_transform._asdict().items()},
        **{f"full_{k}": v for k, v in f1_full._asdict().items()},
        **{f"kvs_{k}": v for k, v in f1_kvs._asdict().items()},
        **{f"keys_{k}": v for k, v in f1_keys._asdict().items()},
        "keys_jaccard": jaccard_keys,
    }


def compute_f1_correctness(
    outputs: list[MetricInput], f1_func: typing.Callable[[SpecType, SpecType], F1Score]
) -> F1Score:
    f1_scores = []
    for output in outputs:
        if output.ground_truth is None:
            continue
        spec_info = output.final_spec_info
        if spec_info is None:
            f1_scores.append(F1Score(float("nan"), float("nan"), float("nan")))
        else:
            scores = f1_func(output.ground_truth, spec_info.spec)
            f1_scores.append(scores)
    return compute_f1_mean(f1_scores)


def compute_spec_metric_mean(outputs: list[MetricInput], func: typing.Callable[[SpecType, SpecType], float]) -> float:
    scores = []
    for output in outputs:
        if output.ground_truth is None:
            continue
        spec_info = output.final_spec_info
        if spec_info is None:
            scores.append(float("nan"))
        else:
            score = func(output.ground_truth, spec_info.spec)
            scores.append(score)
    scores = [x for x in scores if not np.isnan(x)]
    return sum(scores) / len(scores) if len(scores) > 0 else float("nan")


def compute_spec_score_mean(outputs: list[MetricInput]) -> MetricValue:
    scores = []
    for output in outputs:
        score = compute_spec_score(output)
        scores.append(score)
    return sample_mean_confidence_interval(scores)


def lida_sevq_score(scores: list[LIDASelfEvalScore]) -> float:
    """Self-Evaluated Visualization Quality (SEVQ) [LIDA].

    A numeric value from 1-10 (and a rationale) across 6 dimensions - code accuracy, data transformation,
    goal compliance, visualization type, data encoding, and aesthetics.
    The final SEVQ score is the average of the 6 scores [LIDA:App.B].
    """
    if not len(scores) == 6:
        logger.warning(f"Expected 6 LIDA scores, got {len(scores)}: {scores}")
    if len(scores) == 0:
        return float("nan")
    return np.mean([s.score for s in scores if not np.isnan(s.score)], dtype=float)


def lida_self_evaluated_vis_quality(outputs: list[MetricInput]) -> MetricValue:
    scores = []
    for output in outputs:
        lida_scores = output.lida_self_eval_scores
        if lida_scores is None:
            continue
        scores.append(lida_sevq_score(lida_scores))
    return sample_mean_confidence_interval(scores)


def compute_vision_judge_mean(outputs: list[MetricInput]) -> MetricValue:
    scores = []
    for output in outputs:
        if output.vision_judge_score is not None:
            scores.append(output.vision_judge_score)
    return sample_mean_confidence_interval(scores)


def compute_metrics(outputs: list[MetricInput]) -> dict[str, float]:
    ver = visualization_error_rate(outputs)
    resp_error_rate = response_error_rate(outputs)
    empty_plot_rate = empty_plot_error_rate(outputs)
    f1_mark = compute_f1_correctness(outputs, spec_f1_correctness_mark)
    f1_encoding = compute_f1_correctness(outputs, spec_f1_correctness_encoding)
    f1_transform = compute_f1_correctness(outputs, spec_f1_correctness_transform)
    f1_full = compute_f1_correctness(outputs, spec_f1_correctness_full)
    f1_keys = compute_f1_correctness(outputs, spec_f1_correctness_key)
    f1_kvs = compute_f1_correctness(outputs, spec_f1_correctness_key_values)
    jaccard_keys = compute_spec_metric_mean(outputs, spec_jaccard_keys)
    spec_score = compute_spec_score_mean(outputs)
    lida_sevq = lida_self_evaluated_vis_quality(outputs)
    vision_judge = compute_vision_judge_mean(outputs)
    return {
        "count": len(outputs),
        **spec_score.to_dict("spec_score"),
        **ver.to_dict("visualization_error_rate"),
        "response_error_rate": resp_error_rate,
        **empty_plot_rate.to_dict("empty_plot_rate"),
        **lida_sevq.to_dict("lida_sevq"),
        **vision_judge.to_dict("vision_judge"),
        **{f"mark_{k}": v for k, v in f1_mark._asdict().items()},
        **{f"encoding_{k}": v for k, v in f1_encoding._asdict().items()},
        **{f"transform_{k}": v for k, v in f1_transform._asdict().items()},
        **{f"full_{k}": v for k, v in f1_full._asdict().items()},
        **{f"kvs_{k}": v for k, v in f1_kvs._asdict().items()},
        **{f"keys_{k}": v for k, v in f1_keys._asdict().items()},
        "keys_jaccard": jaccard_keys,
        **token_usage_stats(outputs),
    }


def aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-example metrics."""
    results = {}
    transposed = defaultdict(list)
    for m in metrics:
        for k, x in m.items():
            if "_ci" in k or ("_tokens_" in k and not k.endswith("_tokens_mean")):
                continue
            transposed[k].append(x)
    for k, xs in transposed.items():
        if k == "count":
            results[k] = sum(xs)
        elif k in ("spec_score", "lida_sevq", "vision_judge"):
            results.update(**sample_mean_confidence_interval(xs).to_dict(k))
        elif k in ("empty_plot_rate", "visualization_error_rate"):
            results.update(**bootstrap_confidence_interval(xs, np.mean).to_dict(k))
        elif k.endswith("_tokens_mean"):
            results.update(**aggregate_token_counts({k.rstrip("_mean"): xs}))
        else:
            results[k] = sum(xs) / len(xs) if len(xs) > 0 else float("nan")
    return results


if __name__ == "__main__":

    def _main() -> None:
        from pathlib import Path

        from benchmark.vega_chat_benchmark import get_saved_output_paths, read_saved_outputs

        parser = argparse.ArgumentParser()
        parser.add_argument("outputs_path", type=Path, help="Saved outputs path")
        args = parser.parse_args()

        paths = get_saved_output_paths(args.outputs_path)
        for path in paths:
            print(path)
            outputs = read_saved_outputs(path)
            metrics = compute_metrics([out.to_metric_input() for out in outputs])
            pprint.pprint(metrics)

    _main()
