from dataclasses import dataclass
from typing import Any, Self

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from edaplot.data_prompts import DEFAULT_DATA_STRATEGY, DataDescriptionStrategy, get_data_description_prompt
from edaplot.prompt_utils import PromptTag, extract_json_tags


class HeaderQuality(PromptTag):
    OK = "ok"
    UNCLEAR = "unclear"


@dataclass(kw_only=True)
class HeaderAnalyzerResult:
    quality: HeaderQuality
    column_name: str
    rationale: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(quality=HeaderQuality(d["quality"]), column_name=d["column_name"], rationale=d.get("rationale"))


@dataclass(kw_only=True)
class HeaderAnalyzerConfig:
    model_name: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.0
    description_strategy: DataDescriptionStrategy = "main"


@dataclass(kw_only=True)
class HeaderAnalyzerMessage:
    message: BaseMessage
    response: dict[HeaderQuality, list[HeaderAnalyzerResult]] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        if (response := d.pop("response", None)) is not None:
            d["response"] = {
                HeaderQuality(k): [HeaderAnalyzerResult.from_dict(r) for r in results]
                for k, results in response.items()
            }
        return cls(**d)


def get_header_analyzer_prompt(
    df: pd.DataFrame,
    *,
    data_description_strategy: DataDescriptionStrategy = DEFAULT_DATA_STRATEGY,
) -> str:
    data_description = get_data_description_prompt(df, data_description_strategy)

    prompt = f"""\
You are an excellent data analyst. You will analyze the quality of column headers in a dataset.

Group the columns into categories based on their semantic meaning quality. Assume the user is not familiar with the data but has a good understanding of the domain.
  - {HeaderQuality.OK}: names are clear and descriptive. We know what the column is about.
  - {HeaderQuality.UNCLEAR}: names are vague, ambiguous. We don't know what the column is about from its name.

Write your answer in the following format: 
{HeaderQuality.OK.wrap('a json array of column names')}
{HeaderQuality.UNCLEAR.wrap('a json array of objects containing keys: "column_name" and "rationale"')}

## Example output format:
```
{HeaderQuality.OK.wrap('["column_name1", "column_name2"]')}
{HeaderQuality.UNCLEAR.wrap('[{"column_name": "name", "rationale": "a concise single sentence that explains the reasoning behind your answer"}]')}
```

## Dataset format:
{data_description}
"""
    return prompt


def parse_header_analysis(content: str) -> dict[HeaderQuality, list[HeaderAnalyzerResult]]:
    good_results = extract_json_tags(content, HeaderQuality.OK)
    unclear_results = extract_json_tags(content, HeaderQuality.UNCLEAR)
    return {
        HeaderQuality.OK: [HeaderAnalyzerResult(column_name=item, quality=HeaderQuality.OK) for item in good_results],
        HeaderQuality.UNCLEAR: [
            HeaderAnalyzerResult(
                column_name=item["column_name"], rationale=item["rationale"], quality=HeaderQuality.UNCLEAR
            )
            for item in unclear_results
        ],
    }


class HeaderAnalyzer:
    def __init__(self, config: HeaderAnalyzerConfig, model: BaseChatModel) -> None:
        self.config = config
        self.model = model

    async def analyze_header(self, df: pd.DataFrame) -> list[HeaderAnalyzerMessage]:
        prompt = get_header_analyzer_prompt(df, data_description_strategy=self.config.description_strategy)
        messages = [
            HeaderAnalyzerMessage(message=SystemMessage(content=prompt)),
        ]
        response_msg = await self.model.ainvoke([m.message for m in messages])
        assert isinstance(response_msg.content, str)
        messages.append(
            HeaderAnalyzerMessage(message=response_msg, response=parse_header_analysis(response_msg.content))
        )
        return messages

    @classmethod
    def from_config(cls, config: HeaderAnalyzerConfig, api_key: str | None = None) -> Self:
        model = get_model_from_config(config, api_key)
        return cls(config, model)


def get_model_from_config(config: HeaderAnalyzerConfig, api_key: str | None = None) -> BaseChatModel:
    secret_api_key = SecretStr(api_key) if api_key else None
    return ChatOpenAI(model=config.model_name, temperature=config.temperature, api_key=secret_api_key)


def get_header_analyzer_warning(response: dict[HeaderQuality, list[HeaderAnalyzerResult]]) -> tuple[bool, str]:
    unclear_columns = response.get(HeaderQuality.UNCLEAR, [])
    if len(unclear_columns) > 0:
        s = "The following columns may be unclear:\n"
        for item in unclear_columns:
            s += f'- "{item.column_name}": {item.rationale}\n'
        s += (
            f"\n\nConsider providing more context for {'these columns' if len(unclear_columns) > 1 else 'this column'}."
        )
        return True, s
    return False, ""


if __name__ == "__main__":

    def _main() -> None:
        import asyncio
        import pprint

        import vega_datasets

        df = vega_datasets.data.cars()
        df["WTF"] = df[df.columns[0]]  # unclear column name
        analyzer = HeaderAnalyzer.from_config(HeaderAnalyzerConfig())
        output = asyncio.run(analyzer.analyze_header(df))
        pprint.pprint(output)

    _main()
