import logging
import math
import typing
from dataclasses import dataclass
from typing import Literal

import commentjson
import jsonschema
import pandas as pd

from edaplot.data_prompts import DEFAULT_DATA_STRATEGY, DataDescriptionStrategy, get_data_description_prompt
from edaplot.prompt_utils import PromptTag, extract_json_tags, extract_tag_content, print_prompt_helper
from edaplot.spec_utils import SpecType, get_spec_transform_paths

logger = logging.getLogger(__name__)

# TODO refactor into a proper prompt management solution (currently only the system prompt changes)
PromptVersion = Literal["vega_chat_v1", "vega_chat_v2"]

VEGA_LITE_SCHEMA_URL = "https://vega.github.io/schema/vega-lite/v5.json"


class Tag(PromptTag):
    JSON = "json"
    EXPLAIN = "explain"
    EXPLANATION = "explanation"
    RELEVANT = "relevant"
    DATA_EXISTS = "data_exists"
    OUTPUT = "output"


def sys_format_json_str(s: str) -> str:
    return Tag.JSON.wrap(f"\n{s}\n")


def sys_format_json_dict(d: dict) -> str:
    return sys_format_json_str(commentjson.dumps(d, indent=2))


def md_format_json_dict(d: dict) -> str:
    return f"```\n{commentjson.dumps(d, indent=2)}\n```"


def get_system_prompt(
    prompt_version: PromptVersion,
    *,
    df: pd.DataFrame,
    data_description_strategy: DataDescriptionStrategy = DEFAULT_DATA_STRATEGY,
    extra_metadata: str = "",
    language: str | None = "English",
) -> str:
    match prompt_version:
        case "vega_chat_v1":
            return get_system_prompt_v1(
                df,
                data_description_strategy=data_description_strategy,
                extra_metadata=extra_metadata,
                language=language,
            )
        case "vega_chat_v2":
            return get_system_prompt_v2(
                df,
                data_description_strategy=data_description_strategy,
                extra_metadata=extra_metadata,
                language=language,
            )
        case _:
            raise ValueError(f"Unknown prompt version: {prompt_version}")


def get_system_prompt_v1(
    df: pd.DataFrame,
    *,
    data_description_strategy: DataDescriptionStrategy = DEFAULT_DATA_STRATEGY,
    extra_metadata: str = "",
    language: str | None = "English",
) -> str:
    data_description = get_data_description_prompt(df, data_description_strategy)
    data_description = extra_metadata + "\n" + data_description
    # N.B. Reason: sometimes we get Spanish outputs despite using English prompts
    language_spec = language if language is not None else "the same language as the user's utterance"

    # Examples from https://vega.github.io/vega-lite/docs/aggregate.html
    # They are both equivalent, but the first should be preferred
    example_transform_preferred = sys_format_json_dict(
        {"mark": "bar", "encoding": {"x": {"field": "Cylinders"}, "y": {"aggregate": "mean", "field": "Acceleration"}}}
    )
    example_transform_meh = sys_format_json_dict(
        {
            "transform": [
                {"aggregate": [{"op": "mean", "field": "Acceleration", "as": "mean_acc"}], "groupby": ["Cylinders"]}
            ],
            "mark": "bar",
            "encoding": {
                "x": {"field": "Cylinders", "type": "ordinal"},
                "y": {"field": "mean_acc", "type": "quantitative"},
            },
        }
    )

    # From: https://vega.github.io/vega-lite/docs/facet.html
    example_facet_preferred = sys_format_json_dict(
        {
            "mark": "bar",
            "encoding": {
                "x": {"bin": {"maxbins": 15}, "field": "Horsepower", "type": "quantitative"},
                "y": {"aggregate": "count", "type": "quantitative"},
                "row": {"field": "Origin"},
            },
        }
    )
    example_facet_meh = sys_format_json_dict(
        {
            "facet": {"row": {"field": "Origin"}},
            "spec": {
                "mark": "bar",
                "encoding": {
                    "x": {"bin": {"maxbins": 15}, "field": "Horsepower", "type": "quantitative"},
                    "y": {"aggregate": "count", "type": "quantitative"},
                },
            },
        }
    )

    # Normalization example
    normalization_example = sys_format_json_dict(
        {
            "transform": [
                {"aggregate": [{"op": "count", "field": "Origin", "as": "Origin_count"}], "groupby": ["Origin"]},
                {"joinaggregate": [{"op": "sum", "field": "Origin_count", "as": "Origin_count_sum"}]},
                {"calculate": "datum['Origin_count'] / datum['Origin_count_sum']", "as": "Origin_ratio"},
            ],
            "mark": "bar",
            "encoding": {
                "x": {"field": "Origin", "type": "nominal"},
                "y": {"field": "Origin_ratio", "type": "quantitative"},
            },
        }
    )

    # Prompt ideas from:
    # https://github.com/nyanp/chat2plot/blob/main/chat2plot/prompt.py
    # https://arxiv.org/pdf/2401.11255
    # For multi-turn prompts assume only refinements, not a completely new context
    # N.B. using different models may require different prompts (this is for gpt-4o-mini)
    prompt = f"""\
Your task is to generate a valid Vega-Lite chart specification for the given dataset and user utterance delimited by <>. 
The user may want to refine the chart with a follow-up request. The refined chart must include ALL previously given information by the user.
You should do the following step by step, and your response should include both Step 1 and 2:

1. Explain whether filters should be applied to the data, which chart type and columns should be used, and what transformations are necessary to fulfill the user's request. 
  The explanation MUST be in {language_spec}, and be understandable to someone who does not know the JSON schema definition. 
  This text should be enclosed with {Tag.EXPLAIN.open_tag} and {Tag.EXPLAIN.close_tag} tag.
2. Generate Vega-Lite schema-compliant JSON that represents Step 1. 
  This text should be enclosed with {Tag.JSON.open_tag} and {Tag.JSON.close_tag} tag. 

The `data` field must be excluded. The `$schema` field is always set to "{VEGA_LITE_SCHEMA_URL}".
AVOID view-level `transform` unless strictly necessary, in which case `transform` should be placed BEFORE `encoding`. 
DO prefer inlining field transforms inside `encoding` (`bin`, `timeUnit`, `aggregate`, `sort`, and `stack`). 
Here is a GOOD example of an inline aggregation:\n{example_transform_preferred}\n
Here is a valid example of the same view-level transform, which we do NOT prefer:\n{example_transform_meh}\n
If you need to make multiple plots by faceting, use the `column` or `row` encoding channels. Do NOT use the `facet` view-level operator. 
Here is a GOOD example of faceting:\n{example_facet_preferred}\n
Only if the user wants to normalize the data, use the following aggregate-joinaggregate-calculate example:\n{normalization_example}\n

The data has the following format:\n{data_description}
"""
    return prompt


def get_system_prompt_v2(
    df: pd.DataFrame,
    *,
    data_description_strategy: DataDescriptionStrategy = DEFAULT_DATA_STRATEGY,
    extra_metadata: str = "",
    language: str | None = "English",
) -> str:
    data_description = get_data_description_prompt(df, data_description_strategy)
    data_description = extra_metadata + "\n" + data_description
    # N.B. Reason: sometimes we get Spanish outputs despite using English prompts
    language_spec = language if language is not None else "the same language as the user's utterance"

    # Examples from https://vega.github.io/vega-lite/docs/aggregate.html
    # They are both equivalent, but the first should be preferred
    example_transform_preferred = sys_format_json_dict(
        {"mark": "bar", "encoding": {"x": {"field": "Cylinders"}, "y": {"aggregate": "mean", "field": "Acceleration"}}}
    )
    example_transform_meh = sys_format_json_dict(
        {
            "transform": [
                {"aggregate": [{"op": "mean", "field": "Acceleration", "as": "mean_acc"}], "groupby": ["Cylinders"]}
            ],
            "mark": "bar",
            "encoding": {
                "x": {"field": "Cylinders", "type": "ordinal"},
                "y": {"field": "mean_acc", "type": "quantitative"},
            },
        }
    )

    # From: https://vega.github.io/vega-lite/docs/facet.html
    example_facet_preferred = sys_format_json_dict(
        {
            "mark": "bar",
            "encoding": {
                "x": {"bin": {"maxbins": 15}, "field": "Horsepower", "type": "quantitative"},
                "y": {"aggregate": "count", "type": "quantitative"},
                "row": {"field": "Origin"},
            },
        }
    )

    # Normalization example
    normalization_example = sys_format_json_dict(
        {
            "transform": [
                {"aggregate": [{"op": "count", "field": "Origin", "as": "Origin_count"}], "groupby": ["Origin"]},
                {"joinaggregate": [{"op": "sum", "field": "Origin_count", "as": "Origin_count_sum"}]},
                {"calculate": "datum['Origin_count'] / datum['Origin_count_sum']", "as": "Origin_ratio"},
            ],
            "mark": "bar",
            "encoding": {
                "x": {"field": "Origin", "type": "nominal"},
                "y": {"field": "Origin_ratio", "type": "quantitative"},
            },
        }
    )

    # Prompt ideas from:
    # - https://github.com/nyanp/chat2plot/blob/main/chat2plot/prompt.py
    # - https://arxiv.org/pdf/2401.11255
    # - lida
    # For multi-turn prompts assume only refinements, not a completely new context
    # N.B. using different models may require different prompts (this is for gpt-4o-mini)
    # Requirements:
    # - Make sure the data requested exists in the dataset, provide an expected answer if data is not found.
    # - The prompt relies on a dataset header semantic, what if the header has no sense? Can we ask for more info?
    prompt = f"""\
Your task is to generate a valid Vega-Lite chart specification for the given dataset and user utterance delimited by <>. 
The user may want to refine the chart with a follow-up request. The refined chart must include ALL previously given information by the user.

Do the following step by step and always include all steps in your response. All explanations must be in {language_spec}.
1. Explain the visualization goal for the chart you are about to generate. Answer how your visualization goal fulfills the user's request.
  Explain whether filters should be applied to the data, which chart type and columns should be used, and what transformations are necessary to fulfill the user's request.
  Your explanation should be concise. This text should be enclosed within {Tag.EXPLAIN.open_tag} and {Tag.EXPLAIN.close_tag} tags.
2. Determine whether the request is relevant to the given dataset. Provide a brief one sentence rationale. Output format: {Tag.RELEVANT.wrap(f'{{"rationale": "write it here", "{Tag.RELEVANT.value}": true/false}}')}
3. Determine whether ALL the requested data columns exist in the dataset. Provide a brief one sentence rationale. Output format: {Tag.DATA_EXISTS.wrap(f'{{"rationale": "write it here", "{Tag.DATA_EXISTS.value}": true/false}}')}
4. Generate Vega-Lite schema-compliant JSON that fulfills the user's request according to the previous steps. Generate a best-effort valid Vega-Lite JSON even if the user's request cannot be fully fulfilled or the request is not relevant to the dataset.
  This text should be enclosed within {Tag.JSON.open_tag} and {Tag.JSON.close_tag} tags. 

The `data` field must be excluded. The `$schema` field is always set to "{VEGA_LITE_SCHEMA_URL}".
AVOID view-level `transform` unless strictly necessary, in which case `transform` should be placed BEFORE `encoding`. 
DO prefer inlining field transforms inside `encoding` (`bin`, `timeUnit`, `aggregate`, `sort`, and `stack`). 
Here is a GOOD example of an inline aggregation:\n{example_transform_preferred}\n
Here is a valid example of the same view-level transform, which we do NOT prefer:\n{example_transform_meh}\n
If you need to make multiple plots by faceting, use the `column` or `row` encoding channels. Do NOT use the `facet` view-level operator. 
Here is a GOOD example of faceting:\n{example_facet_preferred}\n
Only if the user wants to normalize the data, use the following aggregate-joinaggregate-calculate example:\n{normalization_example}\n

## Example output format:
```
{Tag.EXPLAIN.wrap("A bar chart showing the average stock price over time...")}
{Tag.RELEVANT.wrap(f'{{"rationale": "write it here", "{Tag.RELEVANT}": true}}')}
{Tag.DATA_EXISTS.wrap(f'{{"rationale": "write it here", "{Tag.DATA_EXISTS}": true}}')}
{sys_format_json_str('{"mark": "bar", ...')}
```

## Dataset format:
{data_description}
"""
    return prompt


def get_user_prompt(content: str) -> str:
    return f"<{content}>"


def clear_user_prompt_formatting(content: str) -> str:
    """Undo `get_user_prompt` formatting."""
    return content[1:-1]


def trim_error_message(content: str, max_length: int, sep: str = "\n[trimmed]\n", front_percentage: float = 0.8) -> str:
    if len(content) <= max_length:
        return content
    take_front = math.ceil(max_length * front_percentage)
    take_end = max(0, max_length - take_front)
    if take_end > len(sep):
        take_end -= len(sep)
    else:
        sep = ""
    return content[:take_front] + sep + content[len(content) - take_end :]


def get_error_correction_prompt(error_message: str, *, max_length: int) -> str:
    content = trim_error_message(error_message, max_length=max_length)
    prompt = (
        f"Your response fails with the following error:\n"
        f"```\n{content}\n```\n\n"
        f"Fix the issues in your response. Do not generate the same json again."
    )
    return prompt


def get_manual_error_correction_prompt(content: str) -> str:
    prompt = (
        f"Your response is not a valid Vega-Lite json. "
        f"The problem is:\n{content}\n\n"
        f"Fix the issues in your response. Do not generate the same json again."
    )
    return prompt


def format_schema_validation_error(exc: jsonschema.ValidationError, *, max_length: int) -> str:
    # TODO nicer. See SchemaValidationError for ideas
    content = repr(exc) + "\n" + str(exc)
    return trim_error_message(content, max_length=max_length, front_percentage=0.8)


def format_scenegraph_exception(exc: ValueError, *, max_length: int) -> str:
    content = str(exc)
    content_lines = content.split("\n", maxsplit=2)
    content = "\n".join(content_lines[:2])  # Only the first 1/2 lines are useful
    return trim_error_message(content, max_length=max_length, front_percentage=0.8)


def get_empty_plot_correction_prompt(spec: SpecType, hints: str | None) -> str:
    spec_transforms = [] if spec is None else get_spec_transform_paths(spec)
    has_transforms = len(spec_transforms) > 0
    transforms_prompt = (
        "Verify that the transforms used are really necessary, otherwise remove them. "
        "Make sure the view-level transforms are compatible with the encoding fields.\n"
        if has_transforms
        else ""
    )
    hints_prompt = f"The following hints may be useful:\n{hints}\n" if hints is not None else ""
    prompt = (
        f"Your response is a valid Vega-Lite chart specification, but the plot is empty. "
        f"What is the problem with the chart specification?\n"
        f"{transforms_prompt}"
        f"{hints_prompt}\n"
        f"Fix the issues in your response. Do not generate the same json again."
    )
    return prompt


def get_transform_field_as_missing_prompt(missing_as_fields: list[str]) -> str:
    return (
        f"The following fields appear in the `transform` section, but they are not used in the `encoding` section: {missing_as_fields}.\n"
        f"Correct the json to use the missing fields, or remove them if they are unnecessary. Do not generate the same json again."
    )


def get_transform_in_channel_error_prompt(bad_attributes: list[tuple[str, str]]) -> str:
    return (
        f"The following transforms are not allowed to appear in the `encoding` section: {bad_attributes}.\n"
        f'Fix the error by moving them to the view-level "transform" or correctly inline them if possible. Do not generate the same json again.'
    )


def get_spec_fixed_user_prompt(new_spec: SpecType, user_prompt: str) -> str:
    new_spec_json = md_format_json_dict(new_spec)
    prompt = (
        f"I've made some changes to the Vega-Lite JSON. "
        f"Use the new JSON from now on:\n"
        f"{new_spec_json}"
        f"\n\n{get_user_prompt(user_prompt)}"
    )
    return prompt


def get_new_chart_recommendation_user_prompt() -> str:
    content = "I don't know what the data is about. Show me an interesting plot. Don't show the same plot twice."
    return get_user_prompt(content)


def get_select_chart_prompt(spec: SpecType) -> str:
    spec_json = sys_format_json_dict(spec)
    return f"I am using the following Vega-Lite json:\n{spec_json}"


def get_select_spec_info_prompt(spec: SpecType, *, is_drawable: bool, is_empty_chart: bool) -> str:
    fix_prompt = ""
    if not is_drawable:
        fix_prompt += "The json is invalid. Please fix any errors in your response."
    elif is_empty_chart:
        fix_prompt += "The json draws an empty chart. Please fix the error in your response."
    spec_prompt = get_select_chart_prompt(spec)
    return f"{spec_prompt}\n{fix_prompt}\n"


def extract_response_explanation(content: str) -> str | None:
    explanation_parts = extract_tag_content(content, Tag.EXPLAIN)
    if not explanation_parts:
        explanation_parts = extract_tag_content(content, Tag.EXPLANATION)
    if not explanation_parts:
        return None
    return "\n".join(part.strip() for part in explanation_parts)


def extract_response_json(content: str, tag: PromptTag, *, allow_multiple: bool) -> list[dict]:
    # Sometimes multiple comma-separated JSONs are generated which is not valid JSON.
    # Sometimes multiple <tag> separated JSONs are generated.
    #  Hacky solution: make it into an array to get a valid JSON for parsing
    # LLM sometimes generates JSON with comments, so use the commentjson package instead
    error_msg = (
        f"Make sure to output Vega-Lite schema-compliant JSONs enclosed with {tag.open_tag} and {tag.close_tag} tags."
        if allow_multiple
        else f"Make sure to output a single Vega-Lite schema-compliant JSON enclosed with {tag.open_tag} and {tag.close_tag} tags."
    )

    json_parts = extract_tag_content(content, tag)
    if len(json_parts) == 0:
        logger.error("JSON tags not found")
        raise ValueError(f"Failed to find {tag.open_tag} and {tag.close_tag} tags. {error_msg}")
    if not allow_multiple and len(json_parts) > 1:
        logger.warning("Multiple JSON tags found")

    json_part = ",".join(json_parts)
    try:
        json_dicts = commentjson.loads(f"[{json_part}]")
    except ValueError as e:
        logger.error("extracting model response jsons", exc_info=e)
        raise ValueError(f"Invalid json formatting! {error_msg}")
    if len(json_dicts) == 1:
        if isinstance(json_dicts[0], list):
            json_dicts = json_dicts[0]
    return json_dicts


def extract_response_bool_rationale(content: str, tag: Tag) -> tuple[bool | None, str | None]:
    bool_value = None
    rationale = None
    try:
        json_list = extract_json_tags(content, tag)
        if len(json_list) > 0:
            json_dict = json_list[0]
            bool_value = json_dict.get(tag)
            rationale = json_dict.get("rationale")
    except ValueError:
        pass
    return bool_value, rationale


@dataclass(kw_only=True)
class ModelResponse:
    specs: list[dict]
    explanation: str | None = None
    relevant_request: bool | None = None
    relevant_request_rationale: str | None = None
    data_exists: bool | None = None
    data_exists_rationale: str | None = None


def extract_model_response(content: str, allow_multiple: bool = False) -> ModelResponse:
    """Extract model output by pre-defined tags."""
    specs = extract_response_json(content, Tag.JSON, allow_multiple=allow_multiple)
    explanation = extract_response_explanation(content)
    relevant_request, relevant_request_rationale = extract_response_bool_rationale(content, Tag.RELEVANT)
    data_exists, data_exists_rationale = extract_response_bool_rationale(content, Tag.DATA_EXISTS)
    return ModelResponse(
        specs=specs,
        explanation=explanation,
        relevant_request=relevant_request,
        relevant_request_rationale=relevant_request_rationale,
        data_exists=data_exists,
        data_exists_rationale=data_exists_rationale,
    )


def get_multiple_jsons_error_prompt() -> str:
    return (
        f"Multiple JSONs found! "
        f"Make sure to output a SINGLE Vega-Lite schema-compliant JSON enclosed with {Tag.JSON.open_tag} and {Tag.JSON.close_tag} tags."
    )


if __name__ == "__main__":

    def _print_prompts() -> None:
        import vega_datasets

        df = vega_datasets.data("cars")
        prompt = get_system_prompt(typing.get_args(PromptVersion)[-1], df=df)
        print_prompt_helper("System prompt", prompt)

        prompt = get_empty_plot_correction_prompt({}, "some hint")
        print_prompt_helper("Empty plot correction prompt", prompt)

        prompt = get_error_correction_prompt("some error", max_length=80)
        print_prompt_helper("Exception prompt", prompt)

        prompt = get_spec_fixed_user_prompt({"mark": "bar"}, "My user prompt")
        print_prompt_helper("Spec fixed user prompt", prompt)

    _print_prompts()
