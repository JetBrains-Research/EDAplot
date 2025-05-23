import dataclasses
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Self

import altair as alt
import lida
import pandas as pd
from langchain_core.messages import AIMessage
from lida.components import get_globals_dict as lida_get_globals_dict
from lida.datamodel import ChartExecutorResponse

from benchmark.models.base_models import EvalMessage, EvalModel, EvalModelConfig, EvalModelType
from benchmark.models.utils import remove_code_line
from benchmark.models.vega_chat import VegaChatEvalModel
from edaplot.data_utils import spec_remove_data
from edaplot.vega import MessageType, SpecInfo, process_extracted_specs, validate_spec
from edaplot.vega_chat.vega_chat import ChatSession

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LIDAModelConfig(EvalModelConfig):
    model_type: EvalModelType = EvalModelType.LIDA
    model_name: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.0
    lida_summary: str = "default"
    lida_library: str = "altair"
    lida_max_rows: int | None = 4500
    lida_fix_execute_errors: bool = True  # Our addition! False for original LIDA.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        d["model_type"] = EvalModelType(d["model_type"])
        return cls(**d)


class LIDAEvalModel(EvalModel):
    def __init__(self, config: LIDAModelConfig, df: pd.DataFrame, api_key: str | None = None):
        self.config = config
        self.original_df = df
        self.df = self.preprocess_df(self.original_df)

        self._messages: list[EvalMessage] = []
        self._charts: list[ChartExecutorResponse] = []

        self._textgen_config = lida.TextGenerationConfig(
            n=1, temperature=config.temperature, model=config.model_name, use_cache=True
        )
        self._data_summary: dict[str, Any] | None = None  # lazy init
        self._lida = lida.Manager(lida.llm("openai", api_key=api_key))

    @property
    def messages(self) -> list[EvalMessage]:
        return self._messages

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # See lida.utils.read_dataframe.
        # TODO support column name cleaning (metrics)
        # This is necessary because the lida executor runs altair code which throws a max rows error.
        max_rows = self.config.lida_max_rows
        if max_rows is not None and len(df) > max_rows:
            logger.debug(f"Subsampling lida df to {max_rows} rows")
            return df.sample(max_rows, random_state=0)
        return df

    def _add_message(self, message: EvalMessage) -> None:
        self._messages.append(message)

    def _process_chart_response(self, chart: ChartExecutorResponse) -> EvalMessage:
        if chart.spec is not None:
            spec = process_extracted_specs([chart.spec])
        else:
            spec = None

        response_type = MessageType.AI_RESPONSE_VALID if chart.status else MessageType.AI_RESPONSE_ERROR
        spec_infos = []
        if spec is not None:
            spec_validity = validate_spec(spec, self.df)
            spec_infos.append(
                SpecInfo(
                    spec=spec,
                    is_spec_fixed=False,
                    is_empty_chart=spec_validity.is_empty_scenegraph,
                    is_valid_schema=spec_validity.is_valid_schema,
                    is_drawable=spec_validity.is_valid_scenegraph,
                )
            )
            if not spec_validity.is_valid_scenegraph:
                response_type = MessageType.AI_RESPONSE_ERROR

        # Store chart info as additional message metadata
        chart_dict = dataclasses.asdict(chart)
        chart_dict.pop("code")
        chart_dict.pop("raster")
        ai_message = AIMessage(content="", additional_kwargs=chart_dict)

        return EvalMessage(
            model_type=self.config.model_type,
            message=ai_message,
            message_type=response_type,
            spec_infos=spec_infos,
            explanation=None,
            code=chart.code,
            base64_raster=chart.raster,
        )

    def _fix_false_lida_execute_errors(self, charts: list[ChartExecutorResponse]) -> list[ChartExecutorResponse]:
        """Try to fix errors caused by code execution in `lida.components.executor.ChartExecutor.execute`.

        - `del vega_spec["data"]` after getting a valid alt.Chart
        - inject missing imports
        - fix syntax error `return chart` (missing function body)
        """
        if self.config.lida_library != "altair":
            return charts

        fixable_errors = (
            "KeyError: 'data'",
            "NameError: name 'np' is not defined",
            "SyntaxError: 'return' outside function",
        )
        new_charts = []
        for chart in charts:
            if chart.error is None or all(err not in chart.error["traceback"] for err in fixable_errors):
                new_chart = chart
            else:
                try:
                    code = chart.code
                    ex_locals = lida_get_globals_dict(code, self.df)

                    # N.B. These fixes are not part of LIDA, but we include them since we want a fair (not biased towards our solution)
                    # comparison with our approach.
                    if "NameError: name 'np' is not defined" in chart.error["traceback"]:
                        # The LLM sometimes implicitly assumes `np` is available
                        ex_locals["np"] = importlib.import_module("numpy")
                    if "SyntaxError: 'return' outside function" in chart.error["traceback"]:
                        # Help the LLM if it "forgot" to generate a function body :)
                        code = remove_code_line(code, "return chart")
                        code = remove_code_line(code, "chart = plot(data)")

                    exec(code, ex_locals)
                    alt_chart: alt.Chart = ex_locals["chart"]
                    vega_spec = alt_chart.to_dict()
                    spec_remove_data(vega_spec)  # This is problematic if the code made data transformations...
                    new_chart = ChartExecutorResponse(
                        spec=vega_spec,
                        status=True,
                        raster=None,
                        code=code,
                        library=chart.library,
                    )
                except Exception as e:
                    # Use old chart if some errors happen
                    logger.exception("lida execute code")
                    new_chart = chart
            new_charts.append(new_chart)
        return new_charts

    async def chat(self, prompt: str) -> EvalMessage:
        is_first_turn = len(self._messages) == 0
        if is_first_turn:
            self._data_summary = self._lida.summarize(
                self.df, summary_method=self.config.lida_summary, textgen_config=self._textgen_config
            )

        user_message = ChatSession.create_message(prompt, MessageType.USER)
        self._add_message(VegaChatEvalModel.make_eval_message(user_message))

        if is_first_turn:
            charts = self._lida.visualize(
                summary=self._data_summary,
                library=self.config.lida_library,
                goal=prompt,
                textgen_config=self._textgen_config,
                return_error=True,
            )
        else:
            # chart.code is set even if the last turn was an error
            last_chart = self._charts[-1]
            charts = self._lida.edit(
                code=last_chart.code,
                summary=self._data_summary,
                instructions=[prompt],
                textgen_config=self._textgen_config,
                library=self.config.lida_library,
                return_error=True,
            )
        assert len(charts) == 1, f"Got {len(charts)} charts, expected 1: {charts}"
        if self.config.lida_fix_execute_errors:
            charts = self._fix_false_lida_execute_errors(charts)

        response_message = self._process_chart_response(charts[0])

        self._charts.extend(charts)
        self._add_message(response_message)
        return response_message
