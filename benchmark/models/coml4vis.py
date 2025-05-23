import base64
import io
from dataclasses import dataclass
from typing import Any, Literal, Optional, Self

import pandas as pd
from coml import CoMLAgent, describe_variable  # type: ignore
from coml.prompt_utils import FixContext, GenerateContext  # type: ignore
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from benchmark.models.base_models import EvalMessage, EvalModel, EvalModelConfig, EvalModelType
from benchmark.models.utils import remove_code_line
from benchmark.models.vega_chat import VegaChatEvalModel
from edaplot.vega import MessageType
from edaplot.vega_chat.vega_chat import ChatSession

DUMMY_DATASET_CODE = "dataset = pd.read_csv('data.csv')"


@dataclass(kw_only=True)
class CoMLConfig(EvalModelConfig):
    model_type: EvalModelType = EvalModelType.CoML4Vis
    model_name: str = "gpt-4o-mini-2024-07-18"
    temperature: float = 0.0

    coml_table_format: Literal["coml", "lida"] = "coml"
    coml_library: Literal["matplotlib", "seaborn"] = "matplotlib"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        d["model_type"] = EvalModelType(d["model_type"])
        return cls(**d)


@dataclass(kw_only=True)
class ChartExecutionResult:
    """Response from a visualization execution"""

    status: bool
    code: str
    png: Optional[str] = None
    error_msg: Optional[str] = None


def pre_code(
    df: pd.DataFrame, chart_lib: str, table_format: Literal["coml", "lida"]
) -> tuple[list[str], dict[str, str]]:
    codes = ["import pandas as pd\nimport matplotlib.pyplot as plt\n"]
    if chart_lib == "seaborn":
        codes[-1] += "import seaborn as sns\n"
    codes.append(DUMMY_DATASET_CODE)  # Otherwise SEVQ is confused about the missing dataset.
    variable_descriptions = {
        "dataset": describe_variable(df, dataframe_format=table_format, pandas_description_config=dict(max_rows=10))
    }
    return codes, variable_descriptions


def save_png_b64(plt: Any) -> str:
    buf = io.BytesIO()
    plt.box(False)
    plt.grid(color="lightgray", linestyle="dashed", zorder=-10)
    plt.savefig(buf, format="png", dpi=100, pad_inches=0.2)
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    return plot_data


class CoML4VIS(EvalModel):
    # Code adapted from https://github.com/microsoft/VisEval/blob/main/examples/agent/coml4vis.py

    def __init__(self, config: CoMLConfig, df: pd.DataFrame) -> None:
        self.config = config
        self.df = df

        self._messages: list[EvalMessage] = []
        self._last_context: GenerateContext | FixContext | None = None
        self._execute_history: list[ChartExecutionResult] = []

        llm = ChatOpenAI(model=config.model_name, temperature=config.temperature)
        self.coml = CoMLAgent(llm, num_examples=1, prompt_version=config.coml_library)

    @property
    def messages(self) -> list[EvalMessage]:
        return self._messages

    def generate_code(self, nl_query: str) -> tuple[str, GenerateContext]:
        pre_codes, variable_descriptions = pre_code(self.df, self.config.coml_library, self.config.coml_table_format)
        generating_context = self.coml.generate_code(nl_query, variable_descriptions, pre_codes)
        generate_code = generating_context["answer"]
        return "\n".join(pre_codes) + "\n" + generate_code, generating_context

    def fix_code(self, nl_query: str, context: GenerateContext) -> tuple[str, FixContext]:
        pre_codes, variable_descriptions = pre_code(self.df, self.config.coml_library, self.config.coml_table_format)
        fix_context = self.coml.fix_code(error=None, output=None, hint=nl_query, prev_context=context)
        if fix_context is None:
            fixed_code = context["answer"]  # No change
        else:
            fixed_code = fix_context["interactions"][-1]["code"]
        return "\n".join(pre_codes) + "\n" + fixed_code, fix_context

    def execute_code(self, generated_code: str) -> ChartExecutionResult:
        global_env = {"png": None, "dataset": self.df, "save_png_b64": save_png_b64}
        code = generated_code + "\npng = save_png_b64(plt)"
        code = remove_code_line(code, "plt.show()")
        code = remove_code_line(code, DUMMY_DATASET_CODE)
        try:
            exec(code, global_env)
            png: str | None = global_env["png"]  # type: ignore
            # Save just the original generated_code
            return ChartExecutionResult(code=generated_code, status=True, png=png)
        except Exception as exception_error:
            error_msg = str(exception_error)
            return ChartExecutionResult(code=generated_code, status=False, error_msg=error_msg)

    def _process_chart_response(self, chart: ChartExecutionResult) -> EvalMessage:
        response_type = MessageType.AI_RESPONSE_VALID if chart.status else MessageType.AI_RESPONSE_ERROR
        return EvalMessage(
            model_type=self.config.model_type,
            message=AIMessage(content=""),
            message_type=response_type,
            spec_infos=[],
            explanation=chart.error_msg if chart.error_msg else None,
            code=chart.code,
            base64_raster=chart.png,
        )

    async def chat(self, prompt: str) -> EvalMessage:
        is_first_turn = len(self._messages) == 0

        user_message = ChatSession.create_message(prompt, MessageType.USER)
        self._messages.append(VegaChatEvalModel.make_eval_message(user_message))

        if is_first_turn:
            code, context = self.generate_code(prompt)
        else:
            code, context = self.fix_code(prompt, self._last_context)
        self._last_context = context

        execute_result = self.execute_code(code)
        self._execute_history.append(execute_result)

        response_message = self._process_chart_response(execute_result)
        self._messages.append(response_message)
        return response_message
