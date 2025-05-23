from dataclasses import dataclass

import pandas as pd
from langsmith import traceable

from edaplot.recommend.recommender import ChartRecommender, RecommenderConfig
from edaplot.spec_utils import SpecType
from edaplot.vega import spec_apply_effects, validate_spec
from edaplot.vega_chat.vega_chat import ModelConfig, VegaChat


@dataclass
class QueryChartResult:
    spec: SpecType
    dataframe: pd.DataFrame


def make_interactive_spec(df: pd.DataFrame, spec: SpecType) -> SpecType:
    new_spec = spec_apply_effects(
        spec,
        tooltip_type="encoding",
        replace_tooltip=False,
        interactive_pan_zoom=True,
    )
    # Interactive charts have some limitations and can break otherwise valid charts,
    # so check again and fallback to the old chart if necessary
    # https://vega.github.io/vega-lite/docs/selection.html#current-limitations
    new_validity = validate_spec(new_spec, df)
    if new_validity.is_valid_scenegraph:
        spec = new_spec
    return spec


@traceable
async def generate_query_chart(
    query: str | list[str], df: pd.DataFrame, config: ModelConfig, *, make_interactive: bool = True
) -> QueryChartResult | None:
    """Returns the generated Vega-Lite spec and the preprocessed dataframe that should be used for rendering."""
    if isinstance(query, list):
        query = "\n".join(query)

    model = VegaChat.from_config(config=config, df=df)
    model_out = await model.query(query)

    spec = model_out.spec
    if spec is None or not model_out.is_drawable or model_out.is_empty_chart:
        return None

    preprocessed_df = model.dataframe
    if make_interactive:
        spec = make_interactive_spec(preprocessed_df, spec)
    return QueryChartResult(spec=spec, dataframe=preprocessed_df)


@traceable
async def generate_recommended_charts(
    n_charts: int, df: pd.DataFrame, config: RecommenderConfig
) -> tuple[list[SpecType], pd.DataFrame] | None:
    """Returns the generated Vega-Lite specs and the preprocessed dataframe that should be used for rendering."""
    recommender = ChartRecommender.from_config(config, df)

    await recommender.recommend(n_charts)
    spec_infos = recommender.gather_all_charts()

    specs = []
    for spec_info in spec_infos:
        if spec_info.is_drawable and not spec_info.is_empty_chart:
            specs.append(spec_info.spec)
    if len(specs) == 0:
        return None
    preprocessed_df = recommender.dataframe
    return specs, preprocessed_df
