import typing
from typing import Any, assert_never

import pandas as pd

from benchmark.models.base_models import EvalMessage, EvalModel, EvalModelConfig, EvalModelType
from benchmark.models.coml4vis import CoML4VIS, CoMLConfig
from benchmark.models.lida import LIDAEvalModel, LIDAModelConfig
from benchmark.models.vega_chat import VegaChatEvalConfig, VegaChatEvalModel
from edaplot.data_utils import df_preprocess
from edaplot.vega import MessageType, SpecInfo
from edaplot.vega_chat.vega_chat import MessageInfo, ModelConfig


def get_eval_config_from_dict(config: dict[str, Any]) -> EvalModelConfig:
    if "model_type" not in config:
        # default to support old saved configs
        return VegaChatEvalConfig(model_type=EvalModelType.VEGA_CHAT, model_config=ModelConfig(**config))

    match EvalModelType(config["model_type"]):
        case EvalModelType.VEGA_CHAT:
            return VegaChatEvalConfig.from_dict(config)
        case EvalModelType.LIDA:
            return LIDAModelConfig.from_dict(config)
        case EvalModelType.CoML4Vis:
            return CoMLConfig.from_dict(config)
        case _ as unreachable:
            assert_never(unreachable)


def get_eval_model_from_config(config: EvalModelConfig, df: pd.DataFrame, api_key: str | None = None) -> EvalModel:
    match config:
        case VegaChatEvalConfig() as config_vega_chat:
            return VegaChatEvalModel(config_vega_chat, df, api_key=api_key)
        case LIDAModelConfig() as config_lida:
            return LIDAEvalModel(config_lida, df, api_key=api_key)
        case CoMLConfig() as config_coml:
            return CoML4VIS(config_coml, df)
        case _:
            raise ValueError(f"Unknown model type: {config.model_type}")


def get_eval_message_from_dict(d: dict[str, Any]) -> EvalMessage:
    if "model_type" not in d:
        # old style backwards compatibility
        msg = MessageInfo.from_dict(d)
        eval_msg = VegaChatEvalModel.make_eval_message(msg)
        eval_msg.code = msg.message.additional_kwargs.get("code")
        return eval_msg

    d["model_type"] = EvalModelType(d["model_type"])
    d["message_type"] = MessageType(d["message_type"])
    d["spec_infos"] = [SpecInfo(**info) for info in d["spec_infos"]]
    return EvalMessage(**d)


def preprocess_eval_model_df(df: pd.DataFrame, model_config: EvalModelConfig) -> pd.DataFrame:
    # TODO get df from actual model to avoid having to sync this code
    match model_config.model_type:
        case EvalModelType.VEGA_CHAT:
            conf: VegaChatEvalConfig = typing.cast(VegaChatEvalConfig, model_config)
            normalize_column_names = conf.model_config.data_normalize_column_names
            parse_dates = conf.model_config.data_parse_dates
        case EvalModelType.LIDA | EvalModelType.CoML4Vis:
            normalize_column_names = False
            parse_dates = False
        case _ as unreachable:
            assert_never(unreachable)
    return df_preprocess(df, normalize_column_names=normalize_column_names, parse_dates=parse_dates)
