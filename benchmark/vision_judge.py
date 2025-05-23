import logging
import re
import textwrap
import typing
from dataclasses import dataclass
from typing import Literal

import commentjson
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from benchmark.datasets import ChartLLMDataset, Dataset
from benchmark.models.base_models import EvalMessage, EvalModelConfig
from benchmark.models.eval_models import preprocess_eval_model_df
from edaplot.image_utils import (
    decode_image_base64,
    encode_image_bytes_to_base64,
    get_image_dimensions,
    resize_png_image_to_square,
    vl_to_png_base64,
    vl_to_png_bytes,
)

logger = logging.getLogger(__name__)


@dataclass
class VisionJudgeConfig:
    # gpt-4o is much cheaper for images than gpt-4o-mini: https://openai.com/api/pricing/
    model_name: str = "gpt-4o-2024-11-20"  # "gpt-4o-mini-2024-07-18"
    temperature: float = 0.0
    image_detail: Literal["low", "high", "auto"] = "auto"
    image_scale: float = 2.0
    """Vega-Lite to PNG scale factor."""
    image_resize: bool = True
    """Resize the generated image to be more similar to the ground truth size."""
    prompt_choice: Literal["simple", "labels", "labels_rationale", "criteria"] = "criteria"


@dataclass
class CriteriaPromptResponse:
    score: float = float("nan")
    type: str | None = None
    rationale: str | None = None


@dataclass
class VisionJudgeOutput:
    messages: list[BaseMessage]
    base64_predicted: str | None
    base64_ground_truth: str | None  # Store exact bytes used as input for future reference
    score: float | None = None
    label: str | None = None
    rationale: str | None = None
    criteria: list[CriteriaPromptResponse] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> typing.Self:
        if (criteria := d.get("criteria")) is not None:
            d["criteria"] = [CriteriaPromptResponse(**c) for c in criteria]
        return cls(**d)

    def get_parsed_score(self) -> float | None:
        if self.criteria is not None:
            return compute_vision_judge_criteria_score(self.criteria)
        else:
            return self.score

    def get_is_empty_chart(self) -> bool | None:
        if self.criteria is not None:
            for crit in self.criteria:
                if crit.type == "is_blank":
                    return crit.score == 1
        return None


def get_eval_message_image_base64(
    message: EvalMessage, input_df: pd.DataFrame, model_config: EvalModelConfig, scale: float = 1.0
) -> str | None:
    if message.base64_raster is not None:
        return message.base64_raster
    elif (spec := message.spec) is not None:
        df = preprocess_eval_model_df(input_df, model_config)
        return vl_to_png_base64(spec, df, scale=scale)
    return None


def get_eval_message_image_bytes(
    message: EvalMessage, input_df: pd.DataFrame, model_config: EvalModelConfig, scale: float = 1.0
) -> bytes | None:
    if message.base64_raster is not None:
        return decode_image_base64(message.base64_raster)
    elif (spec := message.spec) is not None:
        df = preprocess_eval_model_df(input_df, model_config)
        return vl_to_png_bytes(spec, df, scale=scale)
    return None


def get_plotting_benchmark_prompt() -> str:
    # Prompt based on plotting_benchmark which is based on MatPlotAgent
    # https://github.com/thunlp/MatPlotAgent/blob/main/evaluation/api_eval.py
    return (
        "You are an excellent judge at evaluating visualization plots between a model generated plot and the ground truth. "
        "You will be giving scores on how well it matches the ground truth plot.\n"
        "The generated plot will be given to you as the first figure. "
        "If the first figure is blank, that means the code failed to generate a figure.\n"
        "Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.\n"
        "Please compare the two figures head to head and rate them.\n"
        "Suppose the second figure has a score of 100, rate the first figure on a scale from 0 to 100.\n"
        "Scoring should be carried out in the following aspect:\n"
        "Plot correctness:\n"
        "Compare closely between the generated plot and the ground truth, the more resemblance the generated plot has compared to the ground truth, the higher the score. "
        "The score should be proportionate to the resemblance between the two plots. "
        "If the plots present the same information but are made in different colors, consider them matching. "
        "Capture the resemblance of the main idea of the plot.\n"
        "Only rate the first figure, the second figure is only for reference.\n"
        "If the first figure is blank, that means the code failed to generate a figure and should get a score of 0.\n"
        "After scoring from the above aspect, please give a final score. "
        "Do not write anything else. "
        "The final score is preceded by the [FINAL SCORE] token. "
        "For example:\n"
        "[FINAL SCORE]: 50"
    )


def parse_plotting_benchmark_prompt_response(content: str) -> float:
    m = re.search(r"\[FINAL SCORE]:\s*?(\d+(?:\.\d+)?)", content, re.MULTILINE | re.IGNORECASE)
    if m is not None:
        return float(m.group(1))
    return float("nan")


def get_matplotagent_prompt(*, include_rationale: bool) -> str:
    # Criteria from the Evaluation guide for humans from the MatPlotAgent paper.
    prompt = textwrap.dedent(
        """
    You are an excellent judge at evaluating visualization plots between a model-generated plot and the ground truth.
    You will be giving scores on how well the generated plot matches the ground truth plot.
    The generated plot will be given to you as the first figure.
    Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.
    Scoring should be carried out based on the following criteria.

    ## Plot Correctness (0-100 points)
    - exact_match (90-100 points): The generated plot is nearly identical to the ground truth, with only minor, negligible differences.
    - high_resemblance (70-89 points): The generated plot closely resembles the ground truth with some small but noticeable differences in data representation or styling.
    - moderate_resemblance (50-69 points): The generated plot has a moderate level of similarity to the ground truth, but there are several noticeable differences that impact the plot’s accuracy or interpretation.
    - low_resemblance (30-49 points): The generated plot shares some similarities with the ground truth but has significant differences that change the overall message or interpretation of the data.
    - poor_match (10-29 points): The generated plot has very little in common with the ground truth, with major discrepancies in data representation.
    - no_resemblance (1-9 points): The generated plot is completely different from the ground truth, with no discernible similarities in data representation.
    - blank (0 points): The generated plot is blank.
    ## Special Considerations
    - Only rate the first figure, the second figure is only for reference.
    - If the plots present the same information but use different colors, axes labels or titles, consider them matching.
    """
    )
    if include_rationale:
        prompt += textwrap.dedent(
            """
        After scoring from the above aspects, please give a final score in jsonl format. You must provide a brief rationale for the final score. Final output example:
        {"rationale": "write it here", "label": "moderate_resemblance", "score": 55}
        """
        )
    else:
        prompt += textwrap.dedent(
            """
        After scoring from the above aspects, please give a final score in jsonl format. Do not write anything else. Final output example:
        {"label": "moderate_resemblance", "score": 55}
        """
        )
    return prompt


class MatPlotAgentResponse(typing.NamedTuple):
    score: float = float("nan")
    label: str | None = None
    rationale: str | None = None


def parse_matplotagent_prompt_response(content: str) -> MatPlotAgentResponse:
    content = content.strip()
    if len(content) == 0:
        return MatPlotAgentResponse()
    try:
        resp = commentjson.loads(content)
    except ValueError:
        return MatPlotAgentResponse()
    if not isinstance(resp, dict):
        return MatPlotAgentResponse()
    try:
        score = float(resp.get("score", "nan"))
    except ValueError:
        score = float("nan")
    label = resp.get("label")
    if label is not None:
        label = str(label)
    rationale = resp.get("rationale")
    if rationale is not None:
        rationale = str(rationale)
    return MatPlotAgentResponse(score, label, rationale)


def get_criteria_prompt() -> str:
    # Our own custom evaluation prompt.
    # Inspiration from LIDA and our previous attempts.
    # See https://huggingface.co/learn/cookbook/en/llm_judge
    prompt = textwrap.dedent(
        f"""
    You are an excellent judge at evaluating visualizations between a model-generated plot and the ground truth plot.
    You will be given the generated plot (first image), the ground_truth plot (second image) and the user's request to generate the first image.
    Scoring should be carried out based on the following criteria.

    - visualization_type: (e.g. bar, line, scatter, etc.)
      - 2 if the visualization type of the generated plot is the same as the ground_truth
      - 1 if the visualization type is different but it doesn't significantly affect the interpretation of the plot compared to the ground_truth (e.g. bar vs line, etc.)
      - 0 otherwise (e.g. line vs donut, etc.)
    - data_encoding: (e.g. what data is used for x, y, color, etc.)
      - 2 if the generated plot encodes the same data as the ground_truth (e.g. same x and y axis, etc.)
      - 1 if the generated plot is missing an encoding channel that is present in the ground_truth (e.g. missing colors or flipped x and y axes, etc.)
      - 0 if the generated plot encodes completely different data to the ground_truth (e.g. price instead of temperature, etc.)
    - data_transformation: (e.g. normalization, aggregation, grouping, etc.)
      - 2 if the data on the generated plot matches the transformations in the ground_truth, or neither image has any transformations applied (e.g. both plots show the average, etc.)
      - 1 if the data is transformed differently to the ground_truth, but the overall interpretation remains similar (e.g. showing median instead of average, etc.). This includes incorrect grouping by time (e.g. by yearmonth instead of by year).
      - 0 otherwise
    - aesthetics: (e.g. colors, titles, legends, etc.)
      - 2 if the generated plot has a similar style as the ground_truth. Colors and axes titles should resemble the ground_truth.
      - 1 if the generated plot has a different style from the ground_truth but that doesn't affect the interpretation of the plot. Colors, text and time units don't have to match.
      - 0 if the generated plot's style has no resemblance to the ground_truth.
    - prompt_compliance:
      - 2 if the generated plot is relevant to the user query, containing all requested information and potentially additional details
      - 1 if the generated plot is partially relevant to the user query, but missing some information requested by the user
      - 0 otherwise
      * This score is independent of the ground_truth!
    - is_blank:
     - 1 if the generated plot is blank, empty, or shows no data except the axes
     - otherwise

    For each criterion, write your score in the following format: `[OUTPUT]: json array of objects with keys "type", "score", "rationale"` 
    The "rationale" should be short and concise.
    
    Example output:
    [OUTPUT]: [{{"type": "visualization_type", "rationale": "write it", "score": 1}}, {{"type": "data_encoding", "rationale": "write it", "score": 1}}, ...]
    """
    )
    return prompt


def parse_criteria_prompt_response(content: str) -> list[CriteriaPromptResponse] | None:
    content = content.strip()
    if len(content) == 0:
        return None
    content = content.removeprefix("[OUTPUT]:").strip()
    try:
        resp = commentjson.loads(content)
    except ValueError:
        return None
    if not isinstance(resp, list):
        return None

    items = []
    for item in resp:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("score", "nan"))
        except ValueError:
            score = float("nan")
        type_ = item.get("type")
        if type_ is not None:
            type_ = str(type_)
        rationale = item.get("rationale")
        if rationale is not None:
            rationale = str(rationale)
        items.append(CriteriaPromptResponse(score, type_, rationale))
    return items


def compute_vision_judge_criteria_score(crits: list[CriteriaPromptResponse]) -> float:
    # Map to (weight, max value)
    crit_info = {
        "visualization_type": (1, 2),
        "data_encoding": (2, 2),
        "data_transformation": (1, 2),
        "aesthetics": (0.75, 2),
        "prompt_compliance": (1.5, 2),
        "is_blank": (1000, 1),  # not inf in case of false positive
    }
    weights = 0.0
    score = 0.0
    for crit in crits:
        if crit.type not in crit_info:
            continue
        weight, max_value = crit_info[crit.type]
        if crit.type == "is_blank":
            # apply penalty only if is_blank=1
            if crit.score == max_value:
                weights += weight
        else:
            score += crit.score / max_value * weight
            weights += weight
    if weights == 0.0:
        return float("nan")
    return score / weights


def remove_image_data_from_message(message: BaseMessage) -> None:
    if isinstance(message, (HumanMessage, SystemMessage)):
        if isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    item["image_url"]["url"] = "[omitted]"


async def run_vision_judge(
    *,
    inp_bytes: bytes | None,
    gt_bytes: bytes | None,
    config: VisionJudgeConfig,
    user_utterance: str | None = None,
) -> VisionJudgeOutput:
    """Judge the predicted image to the ground truth image with a multimodal LLM model."""
    if gt_bytes is None:
        b64_gt = None
    else:
        b64_gt = encode_image_bytes_to_base64(gt_bytes)

    if config.image_resize and inp_bytes is not None and gt_bytes is not None:
        wh_gt = get_image_dimensions(gt_bytes)
        inp_bytes = resize_png_image_to_square(inp_bytes, wh_gt)
    if inp_bytes is None:
        b64_generated = None
    else:
        b64_generated = encode_image_bytes_to_base64(inp_bytes)

    label = None
    rationale = None
    messages: list[BaseMessage] = []
    criteria = None
    score = None
    if b64_generated is None:
        # No predicted image was generated => 0 score.
        score = 0.0
    else:
        model = ChatOpenAI(model=config.model_name, temperature=config.temperature)
        user_prompt = ""
        if config.prompt_choice == "simple":
            system_prompt = get_plotting_benchmark_prompt()
        elif config.prompt_choice == "labels":
            system_prompt = get_matplotagent_prompt(include_rationale=False)
        elif config.prompt_choice == "labels_rationale":
            system_prompt = get_matplotagent_prompt(include_rationale=True)
        elif config.prompt_choice == "criteria":
            assert user_utterance is not None
            system_prompt = get_criteria_prompt()
            user_prompt = f"Here is the user's prompt: {user_utterance}\n"
        else:
            raise ValueError(f"Invalid prompt choice: {config.prompt_choice}")

        # Sometimes the LLM mistakes the gt image for the generated one.
        # Sending the images in separate messages seems to work the same as sending them both in a single message.
        messages.append(SystemMessage(content=system_prompt))
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": f"{user_prompt}Here are the generated and ground truth images:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_generated}",
                            "detail": config.image_detail,
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_gt}",
                            "detail": config.image_detail,
                        },
                    },
                ],
            )
        )
        response = await model.ainvoke(messages)
        messages.append(response)
        response_str = response.content
        assert isinstance(response_str, str)
        if config.prompt_choice == "simple":
            score = parse_plotting_benchmark_prompt_response(response_str)
        elif config.prompt_choice in ("labels", "labels_rationale"):
            score, label, rationale = parse_matplotagent_prompt_response(response_str)
        elif config.prompt_choice == "criteria":
            criteria = parse_criteria_prompt_response(response_str)
        else:
            raise ValueError(f"Invalid prompt choice: {config.prompt_choice}")

    # Avoid image data duplication for lower file size
    for message in messages:
        remove_image_data_from_message(message)

    return VisionJudgeOutput(
        messages=messages,
        base64_predicted=b64_generated,
        base64_ground_truth=b64_gt,
        score=score,
        label=label,
        rationale=rationale,
        criteria=criteria,
    )


async def run_vision_judge_eval_message(
    *,
    eval_message: EvalMessage,
    dataset_id: str,
    dataset: Dataset,
    user_utterance: str,
    config: VisionJudgeConfig,
    model_config: EvalModelConfig,
) -> VisionJudgeOutput:
    """Judge the predicted image to the ground truth image with a multimodal LLM model."""
    # N.B. We will let the model determine if the chart is empty or not.
    dataset_item = dataset[dataset_id]
    gt_image_bytes: bytes | None
    if isinstance(dataset, ChartLLMDataset):
        # chart-llm-gold conveniently already provides png images, so just read them directly
        gt_image_path = dataset.get_png_path(dataset_id)
        gt_image_bytes = gt_image_path.read_bytes()
    else:
        assert dataset_item.ground_truth is not None
        gt_spec = dataset_item.ground_truth
        gt_image_bytes = vl_to_png_bytes(gt_spec, dataset_item.data, scale=config.image_scale)

    generated_image_bytes = get_eval_message_image_bytes(
        eval_message, dataset_item.data, model_config, scale=config.image_scale
    )

    return await run_vision_judge(
        inp_bytes=generated_image_bytes, gt_bytes=gt_image_bytes, config=config, user_utterance=user_utterance
    )
