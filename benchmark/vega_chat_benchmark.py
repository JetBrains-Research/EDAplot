import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Self

import langchain_core.load
from tqdm import tqdm

from benchmark.benchmark_utils import BenchmarkRunConfig, async_run_benchmark_items, get_run_config_path
from benchmark.datasets import Dataset, DatasetItem, NLVCorpusDataset, load_dataset_from_config
from benchmark.lida_self_eval import LIDASelfEvalConfig, LIDASelfEvalOutput, run_lida_self_eval
from benchmark.metrics import MetricInput
from benchmark.models.base_models import EvalMessage, EvalModelConfig
from benchmark.models.eval_models import (
    get_eval_config_from_dict,
    get_eval_message_from_dict,
    get_eval_model_from_config,
)
from benchmark.models.lida import LIDAModelConfig
from benchmark.vision_judge import VisionJudgeConfig, VisionJudgeOutput, run_vision_judge_eval_message
from edaplot.spec_utils import SpecType

logger = logging.getLogger(__name__)


@dataclass
class SavedOutput:
    id: str  # input id == output id == id
    prompt: list[str]
    messages: list[EvalMessage]
    ground_truth: SpecType | None = None
    lida_self_eval: LIDASelfEvalOutput | None = None
    vision_judge: VisionJudgeOutput | None = None
    # TODO save metrics like for evals

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        return langchain_core.load.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> Self:
        # Use langchain's json serialization to deal with BaseMessage
        d = langchain_core.load.loads(s)
        d["messages"] = [get_eval_message_from_dict(m) for m in d["messages"]]
        if (lida_self_eval_dict := d.get("lida_self_eval")) is not None:
            d["lida_self_eval"] = LIDASelfEvalOutput(**lida_self_eval_dict)
        if (vision_judge_dict := d.get("vision_judge")) is not None:
            d["vision_judge"] = VisionJudgeOutput.from_dict(vision_judge_dict)
        return cls(**d)

    @property
    def final_message(self) -> EvalMessage:
        return self.messages[-1]

    def to_metric_input(self) -> MetricInput:
        lida_self_eval_scores = None
        if self.lida_self_eval is not None:
            lida_self_eval_scores = self.lida_self_eval.parse_scores()
        vision_judge_score = None
        vision_judge_is_empty_chart = None
        if self.vision_judge is not None:
            vision_judge_score = self.vision_judge.get_parsed_score()
            vision_judge_is_empty_chart = self.vision_judge.get_is_empty_chart()
        return MetricInput(
            prompts=self.prompt,
            messages=self.messages,
            ground_truth=self.ground_truth,
            lida_self_eval_scores=lida_self_eval_scores,
            vision_judge_score=vision_judge_score,
            vision_judge_is_empty_chart=vision_judge_is_empty_chart,
        )


@dataclass(kw_only=True)
class RunConfig(BenchmarkRunConfig):
    model_config: EvalModelConfig
    dataset_config: dict[str, Any]
    lida_self_eval_config: LIDASelfEvalConfig | None = None
    vision_judge_config: VisionJudgeConfig | None = None
    cli_args: dict[str, Any] | None = None

    def update_cli_args(self, new_args: dict[str, Any] | None) -> None:
        if new_args is not None:
            all_cli_args: dict[str, Any] = self.cli_args if self.cli_args is not None else {}
            all_cli_args.update(new_args)
            self.cli_args = all_cli_args

    @classmethod
    def from_path(cls, path: Path) -> Self:
        config = json.loads(path.read_text())
        config["model_config"] = get_eval_config_from_dict(config["model_config"])
        if (lida_self_eval_config_dict := config.get("lida_self_eval_config")) is not None:
            config["lida_self_eval_config"] = LIDASelfEvalConfig(**lida_self_eval_config_dict)
        if (vision_judge_config_dict := config.get("vision_judge_config")) is not None:
            config["vision_judge_config"] = VisionJudgeConfig(**vision_judge_config_dict)
        return cls(**config)


def read_saved_outputs(path: Path) -> list[SavedOutput]:
    outputs = []
    with open(path) as f:
        for line in f:
            outputs.append(SavedOutput.from_json(line))
    return outputs


def write_outputs(outputs: list[SavedOutput], path: Path, mode: str = "a") -> None:
    with open(path, mode=mode) as f:
        for out in outputs:
            f.write(f"{out.to_json()}\n")


def get_outputs_path(output_dir: Path, run_config: RunConfig) -> Path:
    # N.B. we could just hash the run_config to get a unique name
    dataset_name = run_config.dataset_config["name"]
    if dataset_name == NLVCorpusDataset.name():
        if not run_config.dataset_config["sequential_outputs"]:
            dataset_name = f"{dataset_name}_1p"  # 1p = 1 prompt
        if run_config.dataset_config["sequential_only"]:
            dataset_name = f"{dataset_name}_seq"
    model_name = run_config.model_config.model_type
    return output_dir / f"{model_name}_{dataset_name}.jsonl"


def load_dataset_from_outputs_path(path: Path) -> Dataset:
    config = RunConfig.from_path(get_run_config_path(path))
    return load_dataset_from_config(config.dataset_config)


def get_saved_output_paths(output_dir: Path, recursive: bool = True) -> list[Path]:
    if output_dir.is_file() or output_dir.suffix == ".jsonl":
        return [output_dir]
    if recursive:
        return sorted(output_dir.rglob("*.jsonl"))
    else:
        return sorted(output_dir.glob("*.jsonl"))


def _prepare_incomplete_outputs(
    outputs_path: Path, new_outputs_path: Path, func_is_incomplete: Callable[[SavedOutput], bool]
) -> list[SavedOutput]:
    """Helper to skip existing LLM outputs and return incomplete outputs to be completed."""
    done_output_ids = set()
    if new_outputs_path.exists():
        for out in read_saved_outputs(new_outputs_path):
            done_output_ids.add(out.id)
    else:
        new_outputs_path.touch()
    existing_outputs = read_saved_outputs(outputs_path)
    for out in existing_outputs:
        if not func_is_incomplete(out):
            if out.id not in done_output_ids:
                done_output_ids.add(out.id)
                write_outputs([out], new_outputs_path, mode="a")
    todo_outputs = [out for out in existing_outputs if out.id not in done_output_ids]
    return todo_outputs


async def run_benchmark_dataset_item(
    model_config: EvalModelConfig,
    inp: DatasetItem,
) -> list[EvalMessage]:
    model = get_eval_model_from_config(model_config, inp.data)
    # Run multi-turn prompts in sequence unconditionally
    for prompt in inp.prompt:
        await model.chat(prompt)
    return model.messages


def _multiprocess_run_benchmark_worker(dataset: Dataset, id_: str, model_config: EvalModelConfig) -> SavedOutput:
    inp = dataset[id_]
    model_output = asyncio.run(run_benchmark_dataset_item(model_config, inp))
    return SavedOutput(id=inp.id, prompt=inp.prompt, messages=model_output, ground_truth=inp.ground_truth)


def _multiprocess_run_benchmark(
    dataset: Dataset, model_config: EvalModelConfig, ids: list[str], outputs_path: Path, max_workers: int | None = None
) -> None:
    # Multiprocessing is often faster than async because validating vega specs can take a lot of cpu time
    logger.info(f"Using {max_workers} processes for benchmarking")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = (executor.submit(_multiprocess_run_benchmark_worker, dataset, id_, model_config) for id_ in ids)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(ids)):
            out = future.result()
            write_outputs([out], outputs_path, mode="a")


def run_dataset_benchmark(
    dataset: Dataset,
    model_config: EvalModelConfig,
    *,
    output_dir_or_file: Path,
    max_workers: int = 0,
    max_concurrent: int = 4,
    cli_args: dict[str, Any] | None = None,
) -> Path:
    run_config = RunConfig(dataset_config=dataset.get_config(), model_config=model_config, cli_args=cli_args)
    if output_dir_or_file.is_file() or output_dir_or_file.suffix == ".jsonl":
        outputs_path = output_dir_or_file
    else:
        outputs_path = get_outputs_path(output_dir_or_file, run_config)
    config_path = get_run_config_path(outputs_path)
    run_config.dump_checked(config_path)

    logger.info(f"Saving outputs to: {outputs_path}")
    todo_ids = []
    done_output_ids = set()
    if outputs_path.exists():
        for out in read_saved_outputs(outputs_path):
            done_output_ids.add(out.id)
    else:
        outputs_path.touch()
    for id_ in dataset.iter_ids():
        if id_ not in done_output_ids:
            todo_ids.append(id_)

    cpu_count = os.cpu_count()
    if max_workers == 0 or cpu_count is None:

        async def _aworker(id_: str) -> SavedOutput:
            inp = dataset[id_]
            model_output = await run_benchmark_dataset_item(model_config, inp)
            return SavedOutput(id=inp.id, prompt=inp.prompt, messages=model_output, ground_truth=inp.ground_truth)

        asyncio.run(
            async_run_benchmark_items(
                todo_ids,
                _aworker,
                lambda out: write_outputs([out], outputs_path, mode="a"),
                max_concurrent=max_concurrent,
            )
        )
    else:
        if max_workers < 0:
            max_workers = cpu_count - 1
        _multiprocess_run_benchmark(dataset, model_config, todo_ids, outputs_path, max_workers=max_workers)
    return outputs_path


def update_lida_self_eval_output(out: SavedOutput, eval_config: LIDASelfEvalConfig) -> SavedOutput:
    prompt = "\n".join(out.prompt)

    # For code execution models (LIDA) we can evaluate either the code or the VL spec.
    code = out.final_message.code
    spec = out.final_message.spec
    code_or_spec: str | SpecType | None = spec
    if eval_config.lida_eval_choice == "code":
        code_or_spec = code
        if code is None and spec is not None:
            code_or_spec = spec

    if code_or_spec is not None:
        out.lida_self_eval = run_lida_self_eval(code_or_spec, prompt, eval_config)
    return out


def run_lida_self_eval_on_outputs(
    outputs_path: Path,
    eval_config: LIDASelfEvalConfig,
    *,
    max_workers: int = 0,
    cli_args: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    assert outputs_path.exists(), f"Expected outputs path to exist at {outputs_path}"
    config_path = get_run_config_path(outputs_path)
    assert config_path.exists(), f"Expected run config path to exist at {config_path}"
    run_config = RunConfig.from_path(config_path)
    if isinstance(run_config.model_config, LIDAModelConfig):
        # Set lida library to the stored version
        eval_config.lida_library = run_config.model_config.lida_library
    run_config.lida_self_eval_config = eval_config
    run_config.update_cli_args(cli_args)
    run_config.dump(config_path)

    # Updated outputs are stored in a new file in append mode,
    # and at the end the original file is replaced with the new file.
    new_outputs_path = outputs_path.with_suffix(".lida_self_eval.jsonl")
    logger.info(f"Saving lida self eval outputs to: {new_outputs_path}")

    todo_outputs = _prepare_incomplete_outputs(
        outputs_path, new_outputs_path, lambda out: overwrite or out.lida_self_eval is None
    )

    if max_workers == 0:
        for out in tqdm(todo_outputs):
            out = update_lida_self_eval_output(out, eval_config)
            write_outputs([out], new_outputs_path, mode="a")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = (executor.submit(update_lida_self_eval_output, out, eval_config) for out in todo_outputs)
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(todo_outputs)):
                out = future.result()
                write_outputs([out], new_outputs_path, mode="a")

    # At the end, replace outputs_path with new_outputs_path
    new_outputs_path.replace(outputs_path)
    return outputs_path


def run_vision_judge_on_outputs(
    outputs_path: Path,
    judge_config: VisionJudgeConfig,
    *,
    max_concurrent: int = 4,
    cli_args: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    assert outputs_path.exists(), f"Expected outputs path to exist at {outputs_path}"
    config_path = get_run_config_path(outputs_path)
    assert config_path.exists(), f"Expected run config path to exist at {config_path}"
    run_config = RunConfig.from_path(config_path)
    run_config.vision_judge_config = judge_config
    run_config.update_cli_args(cli_args)
    run_config.dump(config_path)

    # Updated outputs are stored in a new file in append mode,
    # and at the end the original file is replaced with the new file.
    new_outputs_path = outputs_path.with_suffix(".vision_judge.jsonl")
    logger.info(f"Saving vision judge outputs to: {new_outputs_path}")

    dataset = load_dataset_from_config(run_config.dataset_config)
    eval_model_config = run_config.model_config
    todo_outputs = _prepare_incomplete_outputs(
        outputs_path, new_outputs_path, lambda out: overwrite or out.vision_judge is None
    )

    async def _aworker(out: SavedOutput) -> SavedOutput:
        judge_output = await run_vision_judge_eval_message(
            eval_message=out.final_message,
            dataset_id=out.id,
            dataset=dataset,
            user_utterance="\n".join(out.prompt),
            config=judge_config,
            model_config=eval_model_config,
        )
        out.vision_judge = judge_output
        return out

    asyncio.run(
        async_run_benchmark_items(
            todo_outputs,
            _aworker,
            lambda out: write_outputs([out], new_outputs_path, mode="a"),
            max_concurrent=max_concurrent,
        )
    )

    # At the end, replace outputs_path with new_outputs_path
    new_outputs_path.replace(outputs_path)
    return outputs_path
