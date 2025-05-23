import abc
from dataclasses import dataclass
from enum import StrEnum

from langchain_core.messages import BaseMessage

from edaplot.spec_utils import SpecType
from edaplot.vega import MessageType, SpecInfo, VegaMessage


class EvalModelType(StrEnum):
    VEGA_CHAT = "vega_chat"
    LIDA = "lida"
    CoML4Vis = "coml4vis"


@dataclass(kw_only=True)
class EvalModelConfig:
    model_type: EvalModelType


@dataclass(kw_only=True)
class EvalMessage(abc.ABC):
    # Wrapper for all model output types. Must be serializable.
    model_type: EvalModelType
    message: BaseMessage
    message_type: MessageType
    spec_infos: list[SpecInfo]
    explanation: str | None = None
    code: str | None = None
    base64_raster: str | None = None  # For matplotlib images (from LIDA)

    @property
    def spec(self) -> SpecType | None:
        if len(self.spec_infos) > 0:
            return self.spec_infos[0].spec
        return None

    def to_vega_message(self) -> VegaMessage:
        return VegaMessage(
            message=self.message,
            message_type=self.message_type,
            spec_infos=self.spec_infos,
            explanation=self.explanation,
        )


class EvalModel(abc.ABC):
    """Minimal model interface for running benchmarks."""

    @property
    @abc.abstractmethod
    def messages(self) -> list[EvalMessage]:
        pass

    @abc.abstractmethod
    async def chat(self, prompt: str) -> EvalMessage:
        pass
