from dataclasses import dataclass
from typing import Any, Self

import pandas as pd

from benchmark.models.base_models import EvalMessage, EvalModel, EvalModelConfig, EvalModelType
from edaplot.vega_chat.vega_chat import MessageInfo, ModelConfig, VegaChat


@dataclass(kw_only=True)
class VegaChatEvalConfig(EvalModelConfig):
    model_config: ModelConfig
    model_type: EvalModelType = EvalModelType.VEGA_CHAT

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        d["model_type"] = EvalModelType(d["model_type"])
        d["model_config"] = ModelConfig.from_dict(d["model_config"])
        return cls(**d)


class VegaChatEvalModel(EvalModel):
    def __init__(self, config: VegaChatEvalConfig, df: pd.DataFrame, api_key: str | None = None):
        self.model = VegaChat.from_config(config.model_config, df, api_key=api_key)

    @staticmethod
    def make_eval_message(msg: MessageInfo) -> EvalMessage:
        vega_msg = msg.to_vega_message()
        return EvalMessage(
            model_type=EvalModelType.VEGA_CHAT,
            message=vega_msg.message,
            message_type=vega_msg.message_type,
            spec_infos=vega_msg.spec_infos,
            explanation=vega_msg.explanation,
        )

    @property
    def messages(self) -> list[EvalMessage]:
        return [self.make_eval_message(m) for m in self.model.session.messages]

    async def chat(self, prompt: str) -> EvalMessage:
        await self.model.query(prompt)
        return self.make_eval_message(self.model.session.last_message)
