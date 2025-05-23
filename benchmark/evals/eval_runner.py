import asyncio
import json
import logging
import pprint
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Self, assert_never

import pandas as pd
from tqdm import tqdm

from benchmark.benchmark_utils import (
    BenchmarkRunConfig,
    async_run_benchmark_items,
    get_run_config_path,
    get_timestamp_path,
)
from benchmark.evals.eval_checks import (
    CheckColors,
    CheckGroundTruth,
    CheckHasField,
    CheckHasFieldKwargs,
    CheckHeaderAnalyzer,
    CheckMark,
    CheckRequestAnalyzer,
    CheckTransform,
)
from benchmark.evals.eval_types import (
    ActionOutput,
    ActionType,
    CheckType,
    EvalCheckResult,
    EvalInput,
    EvalOutput,
    InputAction,
)
from benchmark.metrics import aggregate_metrics
from benchmark.vega_chat_benchmark import get_saved_output_paths
from edaplot.app_state import AppState
from edaplot.paths import PATH_EVALS
from edaplot.spec_utils import SpecType
from edaplot.vega import SpecInfo
from edaplot.vega_chat.vega_chat import ModelConfig

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RunConfig(BenchmarkRunConfig):
    chat_model_config: ModelConfig

    @classmethod
    def from_path(cls, path: Path) -> Self:
        config = json.loads(path.read_text())
        config["chat_model_config"] = ModelConfig.from_dict(config["chat_model_config"])
        return cls(**config)


def get_outputs_path(output_dir: Path, run_config: RunConfig) -> Path:
    return get_timestamp_path(output_dir, run_config.timestamp)


def read_saved_outputs(path: Path) -> dict[str, EvalOutput]:
    outputs = {}
    with open(path) as f:
        for line in f:
            output = EvalOutput.from_json(line)
            outputs[output.input.id] = output
    return outputs


def write_outputs(outputs: list[EvalOutput], path: Path, mode: str = "a") -> None:
    with open(path, mode=mode) as f:
        for out in outputs:
            f.write(f"{out.to_json()}\n")


def read_eval_inputs(path: Path = PATH_EVALS) -> list[EvalInput]:
    return [EvalInput.from_yaml(p) for p in sorted(path.glob("*.yaml"))]


def check_eval_action_output(
    df: pd.DataFrame, actions: list[InputAction], action_outputs: list[ActionOutput], action_idx: int
) -> EvalCheckResult:
    results: EvalCheckResult = {}
    check_actions = actions[: action_idx + 1]
    check_action_outputs = action_outputs[: action_idx + 1]
    for check in actions[action_idx].checks:
        match check.check_type:
            case CheckType.MARK:
                check_mark = CheckMark(**check.check_kwargs)
                results[check.check_type] = check_mark(df, check_actions, check_action_outputs)
            case CheckType.GROUND_TRUTH:
                check_ground_truth = CheckGroundTruth(**check.check_kwargs)
                results[check.check_type] = check_ground_truth(df, check_actions, check_action_outputs)
            case CheckType.COLORS:
                check_colors = CheckColors(**check.check_kwargs)
                results[check.check_type] = check_colors(df, check_actions, check_action_outputs)
            case CheckType.HAS_FIELD:
                kwargs = [CheckHasFieldKwargs(**kwargs) for kwargs in check.check_kwargs["fields"]]
                check_has_field = CheckHasField(kwargs)
                results[check.check_type] = check_has_field(df, check_actions, check_action_outputs)
            case CheckType.TRANSFORM:
                check_transform = CheckTransform(**check.check_kwargs)
                results[check.check_type] = check_transform(df, check_actions, check_action_outputs)
            case CheckType.REQUEST_ANALYZER:
                check_request_analyzer = CheckRequestAnalyzer(should_warn=check.check_kwargs["should_warn"])
                results[check.check_type] = check_request_analyzer(df, check_actions, check_action_outputs)
            case CheckType.HEADER_ANALYZER:
                check_header_analyzer = CheckHeaderAnalyzer(unclear=check.check_kwargs["unclear"])
                results[check.check_type] = check_header_analyzer(df, check_actions, check_action_outputs)
            case _ as unreachable:
                assert_never(unreachable)
    return results


def check_eval_output(eval_input: EvalInput, eval_output: EvalOutput) -> list[EvalCheckResult]:
    df = eval_input.load_dataframe()
    results = []
    for action_idx, action in enumerate(eval_input.actions):
        res = check_eval_action_output(df, eval_input.actions, eval_output.action_outputs, action_idx)
        results.append(res)
    return results


def update_eval_checks(eval_inputs: list[EvalInput], outputs_path: Path) -> None:
    saved_outputs = read_saved_outputs(outputs_path)
    for inp in eval_inputs:
        if inp.id in saved_outputs:
            eval_output = saved_outputs[inp.id]
            eval_output.input = inp  # Assume only checks can change!
            eval_output.check_results = check_eval_output(inp, eval_output)
    logger.info(f"Writing eval outputs with checks to {outputs_path}...")
    write_outputs(list(saved_outputs.values()), outputs_path, mode="w")


def flatten_check_results(check_results: list[EvalCheckResult]) -> list[dict[str, float]]:
    results = []
    for res in check_results:
        metrics = {}
        for check_metrics in res.values():
            metrics.update(check_metrics)
        results.append(metrics)
    return results


def check_results_to_df(check_results: list[EvalCheckResult]) -> pd.DataFrame:
    results = {}
    for i, action_results in enumerate(flatten_check_results(check_results)):
        results[i] = action_results
    return pd.DataFrame.from_dict(results, orient="index")


def compute_check_score(check_results: EvalCheckResult) -> float:
    """Single score that combines all individual check results."""
    check_results = {k: v for k, v in check_results.items() if len(v) > 0}  # skip missing check results
    check_score = 0.0
    weights = 0.0
    for check_type, results in check_results.items():
        match check_type:
            case CheckType.MARK:
                w = 0.5  # too easy
                score = results["check_mark"]
            case CheckType.GROUND_TRUTH:
                w = 1.0
                score = results["spec_score"]
            case CheckType.COLORS:
                w = 1.0
                score = results["check_colors"]
            case CheckType.HAS_FIELD:
                w = 1.0
                score = results["check_has_fields"]
            case CheckType.TRANSFORM:
                w = 1.0
                score = results["check_transform"]
            case CheckType.REQUEST_ANALYZER:
                w = 1.0
                score = results["check_request_analyzer"]
            case CheckType.HEADER_ANALYZER:
                w = 1.0
                score = results["check_header_analyzer"]
            case _ as unreachable:
                assert_never(unreachable)
        assert 0.0 <= score <= 1.0
        weights += w
        check_score += w * score
    return check_score / weights


def aggregate_check_results(check_results: list[EvalCheckResult]) -> EvalCheckResult:
    results = {}
    transposed = defaultdict(list)
    for m in check_results:
        for k, x in m.items():
            transposed[k].append(x)
    for k, metrics in transposed.items():
        results[k] = aggregate_metrics(metrics)
    return results


def compute_checks_metrics(check_results: list[EvalCheckResult]) -> dict[str, float]:
    aggr_results = aggregate_check_results(check_results)
    metrics = {k: v for inner_dict in aggr_results.values() for k, v in inner_dict.items()}
    metrics["check_count"] = sum(len(checks) for checks in check_results)
    metrics["check_score"] = compute_check_score(aggr_results)
    return metrics


def print_eval_results(outputs_path: Path) -> None:
    outputs = read_saved_outputs(outputs_path)
    for out in outputs.values():
        print(f"id={out.input.id}")
        for i, (action, check_results) in enumerate(zip(out.input.actions, out.check_results)):
            print(f" [{i}] Action: {action.action_type}: {action.action_kwargs}")
            print(f" [{i}] Checks:")
            pprint.pprint(check_results)
        print()

    print("Total results:")
    all_check_results = []
    for out in outputs.values():
        all_check_results.extend(out.check_results)
    # results = aggregate_check_results(all_check_results)
    results = compute_checks_metrics(all_check_results)
    pprint.pprint(results)


async def run_eval_action(app_state: AppState, action: InputAction, n_chat_messages: int) -> ActionOutput:
    match action.action_type:
        case ActionType.USER_UTTERANCE:
            user_input = action.action_kwargs["user_utterance"]
            app_state.schedule_chat(user_input, is_prompt_formatted=False)
            await app_state.run_chat()
        case ActionType.SELECT_CHART:
            spec: SpecType = action.action_kwargs["spec"]
            app_state.set_recommended_charts(spec_infos=[SpecInfo.from_valid(spec)], selected_idx=0)
        case ActionType.HEADER_ANALYZER:
            app_state.schedule_header_analyzer()
            await app_state.run_header_analyzer()
        case _ as unreachable:
            assert_never(unreachable)

    vega_chat_messages = None
    request_analyzer_history = None
    header_analyzer_messages = (
        app_state.header_analyzer_messages if action.action_type == ActionType.HEADER_ANALYZER else None
    )
    if action.action_type != ActionType.HEADER_ANALYZER:
        chat_messages = app_state.get_chat_messages()
        vega_chat_messages = [m.vega_chat_message for m in chat_messages[n_chat_messages:]]
        request_analyzer_history = [m.request_analyzer_output for m in chat_messages[n_chat_messages:]]
    return ActionOutput(
        vega_chat_messages=vega_chat_messages,
        request_analyzer_history=request_analyzer_history,
        header_analyzer_history=header_analyzer_messages,
    )


async def run_eval_input(
    chat_model_config: ModelConfig, eval_input: EvalInput, *, perform_checks: bool = False
) -> EvalOutput:
    df = eval_input.load_dataframe()
    app_state = AppState()
    app_state.init_state(df=df, chat_config=chat_model_config)
    app_state.set_request_analyzer_enabled(True)  # TODO configurable
    n_chat_messages = 0  # 0 to include the system prompt
    check_results = []
    action_outputs = []
    for action_idx, action in enumerate(eval_input.actions):
        action_output = await run_eval_action(app_state, action, n_chat_messages)
        action_outputs.append(action_output)
        if perform_checks:
            check = check_eval_action_output(df, eval_input.actions, action_outputs, action_idx)
            check_results.append(check)
        n_chat_messages += len(action_output.vega_chat_messages) if action_output.vega_chat_messages is not None else 0
    return EvalOutput(input=eval_input, action_outputs=action_outputs, check_results=check_results)


def run_evals(
    eval_inputs: list[EvalInput],
    chat_model_config: ModelConfig,
    *,
    output_dir_or_file: Path,
    max_concurrent: int = 4,
) -> Path:
    run_config = RunConfig(chat_model_config=chat_model_config)
    if output_dir_or_file.is_file() or output_dir_or_file.suffix == ".jsonl":
        outputs_path = output_dir_or_file
    else:
        outputs_path = get_outputs_path(output_dir_or_file, run_config)
    config_path = get_run_config_path(outputs_path)
    run_config.dump_checked(config_path)

    logger.info(f"Saving outputs to: {outputs_path}")
    outputs = {}
    if outputs_path.exists():
        outputs = read_saved_outputs(outputs_path)
    outputs_path.touch(exist_ok=True)

    inputs = []
    for inp in eval_inputs:
        if inp.id not in outputs:
            inputs.append(inp)

    async def worker(eval_input: EvalInput) -> EvalOutput:
        return await run_eval_input(chat_model_config, eval_input)

    asyncio.run(
        async_run_benchmark_items(
            inputs, worker, lambda out: write_outputs([out], outputs_path, mode="a"), max_concurrent=max_concurrent
        )
    )

    # Run checks after all messages have been safely written
    update_eval_checks(eval_inputs, outputs_path)

    return outputs_path


def update_all_eval_checks(outputs_path: Path) -> None:
    """Update the stored eval check results from the given directory or single file."""
    eval_inputs = read_eval_inputs()
    paths = get_saved_output_paths(outputs_path)
    for path in (pbar := tqdm(paths)):
        pbar.set_description(str(path))
        update_eval_checks(eval_inputs, path)
        # print_eval_results(path)
