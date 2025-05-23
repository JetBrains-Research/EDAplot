import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import langchain_core.load
import yaml

from benchmark.benchmark_utils import (
    BenchmarkRunConfig,
    async_run_benchmark_items,
    get_run_config_path,
    get_timestamp_path,
)
from benchmark.vision_judge import VisionJudgeConfig, VisionJudgeOutput, run_vision_judge
from edaplot.paths import PATH_VISION_JUDGE_BENCHMARK_DIR, PATH_VISION_JUDGE_BENCHMARK_IMAGES

logger = logging.getLogger(__name__)


@dataclass
class VisionJudgeBenchmarkInput:
    id: str
    utterance: str

    @property
    def ground_truth_image_path(self) -> Path:
        return PATH_VISION_JUDGE_BENCHMARK_IMAGES / f"{self.id}_gt.png"

    @property
    def input_image_path(self) -> Path:
        return PATH_VISION_JUDGE_BENCHMARK_IMAGES / f"{self.id}_gen.png"


@dataclass
class VisionJudgeBenchmarkOutput:
    input: VisionJudgeBenchmarkInput
    output: VisionJudgeOutput

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        return langchain_core.load.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> Self:
        d = langchain_core.load.loads(s)
        d["input"] = VisionJudgeBenchmarkInput(**d["input"])
        d["output"] = VisionJudgeOutput.from_dict(d["output"])
        return cls(**d)


@dataclass(kw_only=True)
class RunConfig(BenchmarkRunConfig):
    vision_judge_config: VisionJudgeConfig | None = None
    inputs_path: str | None = None
    cli_args: dict[str, Any] | None = None

    @classmethod
    def from_path(cls, path: Path) -> Self:
        config = json.loads(path.read_text())
        config["vision_judge_config"] = VisionJudgeConfig(**config["vision_judge_config"])
        return cls(**config)


def read_benchmark_inputs(
    path: Path = PATH_VISION_JUDGE_BENCHMARK_DIR / "benchmark.yaml",
) -> list[VisionJudgeBenchmarkInput]:
    with open(path, "r") as f:
        inputs = yaml.safe_load(f)
    return [VisionJudgeBenchmarkInput(**inp) for inp in inputs]


def get_outputs_path(output_dir: Path, run_config: RunConfig) -> Path:
    return get_timestamp_path(output_dir, run_config.timestamp)


def read_saved_outputs(path: Path) -> dict[str, VisionJudgeBenchmarkOutput]:
    outputs = {}
    with open(path) as f:
        for line in f:
            output = VisionJudgeBenchmarkOutput.from_json(line)
            outputs[output.input.id] = output
    return outputs


def write_outputs(outputs: list[VisionJudgeBenchmarkOutput], path: Path, mode: str = "a") -> None:
    with open(path, mode=mode) as f:
        for out in outputs:
            f.write(f"{out.to_json()}\n")


def print_outputs(outputs_path: Path) -> None:
    outputs = read_saved_outputs(outputs_path)
    for id_, out in outputs.items():
        print("* ", id_)
        print("> ", out.output.messages[-1].content)


def run_vision_judge_benchmark(
    inputs_path: Path,
    output_dir_or_file: Path,
    judge_config: VisionJudgeConfig,
    *,
    max_concurrent: int = 4,
    cli_args: dict[str, Any] | None = None,
) -> Path:
    run_config = RunConfig(vision_judge_config=judge_config, cli_args=cli_args, inputs_path=inputs_path.name)
    if output_dir_or_file.is_file() or output_dir_or_file.suffix == ".jsonl":
        outputs_path = output_dir_or_file
    else:
        outputs_path = get_outputs_path(output_dir_or_file, run_config)
    config_path = get_run_config_path(outputs_path)
    run_config.dump_checked(config_path)
    logger.info(f"Saving outputs to: {outputs_path}")

    done_ids = set()
    if outputs_path.exists():
        for inp_id in read_saved_outputs(outputs_path):
            done_ids.add(inp_id)
    else:
        outputs_path.touch()
    inputs = read_benchmark_inputs(inputs_path)
    todo_inputs = []
    for inp in inputs:
        if inp.id not in done_ids:
            todo_inputs.append(inp)

    async def worker(inp: VisionJudgeBenchmarkInput) -> VisionJudgeBenchmarkOutput:
        inp_bytes = inp.input_image_path.read_bytes()
        gt_bytes = inp.ground_truth_image_path.read_bytes()
        judge_output = await run_vision_judge(
            inp_bytes=inp_bytes, gt_bytes=gt_bytes, config=judge_config, user_utterance=inp.utterance
        )
        return VisionJudgeBenchmarkOutput(input=inp, output=judge_output)

    asyncio.run(
        async_run_benchmark_items(
            todo_inputs, worker, lambda out: write_outputs([out], outputs_path, mode="a"), max_concurrent=max_concurrent
        )
    )

    return outputs_path
