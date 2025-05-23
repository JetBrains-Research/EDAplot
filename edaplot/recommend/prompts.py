from dataclasses import dataclass

import pandas as pd

from edaplot.data_prompts import DEFAULT_DATA_STRATEGY, DataDescriptionStrategy, get_data_description_prompt
from edaplot.prompt_utils import PromptTag, print_prompt_helper
from edaplot.vega_chat.prompts import (
    VEGA_LITE_SCHEMA_URL,
    extract_response_explanation,
    extract_response_json,
    sys_format_json_dict,
)


class Tag(PromptTag):
    JSON = "json"
    EXPLAIN = "explain"
    OUTPUT = "output"


def get_system_prompt(
    df: pd.DataFrame,
    *,
    data_description_strategy: DataDescriptionStrategy = DEFAULT_DATA_STRATEGY,
) -> str:
    # Prompt ideas from:
    # https://github.com/microsoft/lida/blob/main/lida/components/viz/vizrecommender.py

    data_description = get_data_description_prompt(df, data_description_strategy)

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

    prompt = (
        f"You are a helpful assistant highly skilled in recommending a DIVERSE set of valid Vega-Lite chart specifications for the given dataset. "
        f"Your task is to recommend visualizations that a user may be interested in. "
        f"Your recommendations may consider different types of valid data aggregations, chart types, clearer ways of displaying information and use different variables from the data summary. "
        # Your goal is to generate the best exploratory data analysis visualizations in terms of representation accuracy, visualization clarity, avoidance of anti-patterns, effective use of visualization techniques, insightfulness, attention to detail.
        f"\n\nThe data has the following format:\n{data_description}\n\n"
        f"You should do the following step by step, and your response should include both 1 and 2:\n"
        f"1. Select and list the most valuable columns for visualization based on column semantics and values. "
        f"Visualizations SHOULD provide valuable insights for exploratory data analysis. "
        f"This text should be enclosed with {Tag.EXPLAIN.open_tag} and {Tag.EXPLAIN.close_tag} tags.\n"
        f"2. Generate as many Vega-Lite schema-compliant JSONs that represent 1. as desired by the user. "
        f"Each JSON must be placed in its own snippet enclosed with {Tag.JSON.open_tag} and {Tag.JSON.close_tag} tags. "
        f"The `data` field must be excluded. "
        f'The `$schema` field is always set to "{VEGA_LITE_SCHEMA_URL}". '
        f"AVOID view-level `transform` unless strictly necessary, in which case `transform` should be placed BEFORE `encoding`. "
        f"DO prefer inlining field transforms inside `encoding` (`bin`, `timeUnit`, `aggregate`, `sort`, and `stack`). "
        f"Here is a GOOD example of an inline aggregation:\n{example_transform_preferred}\n"
        f"Here is a valid example of the same view-level transform, which we do NOT prefer:\n{example_transform_meh}\n"
        f"If you need to make multiple plots by faceting, use the `column` or `row` encoding channels. "
        f"Do NOT use the `facet` view-level operator. Here is a GOOD example of faceting:\n{example_facet_preferred}\n"
        # The generator doesn't make mistakes, but the recommender does...
        f'"mark" property value MUST BE one of: "boxplot", "errorbar", "errorband", "arc", "area", "bar", "image", "line", "point", "rect", "rule", "text", "tick", "trail", "circle", "square", "geoshape". '
        f'Other values are NOT ALLOWED for "mark" property.'
    )
    return prompt


def get_user_prompt(n_charts: int) -> str:
    return f"Recommend {n_charts} (N={n_charts}) charts."


def get_multiple_replies_prompt(replies: list[str | None]) -> str:
    reply = ""
    for i, spec_reply in enumerate(replies):
        if spec_reply is None:
            reply += f"Chart {i}: Good.\n\n"
        else:
            reply += f"Chart {i}: {spec_reply}\n\n"
    return reply.rstrip()


@dataclass(kw_only=True)
class ModelResponse:
    specs: list[dict]
    explanation: str | None = None


def extract_model_response(content: str, allow_multiple: bool = False) -> ModelResponse:
    """Extract model output by pre-defined tags."""
    specs = extract_response_json(content, Tag.JSON, allow_multiple=allow_multiple)
    explanation = extract_response_explanation(content)
    return ModelResponse(
        specs=specs,
        explanation=explanation,
    )


if __name__ == "__main__":

    def _print_prompts() -> None:
        import vega_datasets

        df = vega_datasets.data("cars")
        prompt = get_system_prompt(df)
        print_prompt_helper("System prompt", prompt)

        prompt = get_user_prompt(3)
        print_prompt_helper("User prompt", prompt)

        prompt = get_multiple_replies_prompt(["First", None, "Third\nhas\nlines"])
        print_prompt_helper("Multiple replies prompt", prompt)

    _print_prompts()
