import asyncio
import dataclasses
import itertools
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, assert_never

import langchain_core.load
import pandas as pd

from benchmark.benchmark_utils import (
    BenchmarkRunConfig,
    async_run_benchmark_items,
    get_run_config_path,
    get_timestamp_path,
)
from benchmark.datasets import Dataset, DatasetItem
from benchmark.metrics import sample_mean_confidence_interval
from edaplot.request_analyzer.request_analyzer import (
    RequestAnalyzer,
    RequestAnalyzerConfig,
    RequestAnalyzerOutput,
    RequestTypeMessage,
    get_data_availability_warning,
    get_request_type_missing_data_warning,
)
from edaplot.spec_utils import SpecType, get_spec_field

logger = logging.getLogger(__name__)


class RequestAnalyzerBenchmarkType(StrEnum):
    NORMAL = "normal"
    DROP_ONE_GROUND_TRUTH = "drop_one_ground_truth"
    DROP_ALL_GROUND_TRUTH = "drop_all_ground_truth"


@dataclass(kw_only=True)
class RequestAnalyzerBenchmarkInput:
    id: str
    benchmark_type: RequestAnalyzerBenchmarkType
    prompt: list[str]
    ground_truth: SpecType | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        d["benchmark_type"] = RequestAnalyzerBenchmarkType(d["benchmark_type"])
        return cls(**d)


@dataclass(kw_only=True)
class RequestAnalyzerBenchmarkOutput:
    input: RequestAnalyzerBenchmarkInput
    response_history: list[RequestAnalyzerOutput]
    dropped_fields: list[str] | None = None

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        d = langchain_core.load.loads(json_str)
        d["input"] = RequestAnalyzerBenchmarkInput.from_dict(d["input"])
        if isinstance(d["response_history"][0], list):  # old format
            d["response_history"] = [
                RequestAnalyzerOutput(
                    request_type=[RequestTypeMessage.from_dict(m) for m in ms],
                )
                for ms in d["response_history"]
            ]
        else:
            d["response_history"] = [RequestAnalyzerOutput.from_dict(m) for m in d["response_history"]]
        return cls(**d)

    def to_json(self) -> str:
        return langchain_core.load.dumps(dataclasses.asdict(self))


@dataclass(kw_only=True)
class RunConfig(BenchmarkRunConfig):
    dataset_config: dict[str, Any]
    request_analyzer_config: RequestAnalyzerConfig
    cli_args: dict[str, Any] | None = None

    @classmethod
    def from_path(cls, path: Path) -> Self:
        config = json.loads(path.read_text())
        config["request_analyzer_config"] = RequestAnalyzerConfig(**config["request_analyzer_config"])
        return cls(**config)


def get_outputs_path(output_dir: Path, run_config: RunConfig) -> Path:
    return get_timestamp_path(output_dir, run_config.timestamp)


def read_saved_outputs(path: Path) -> dict[str, RequestAnalyzerBenchmarkOutput]:
    outputs = {}
    with open(path) as f:
        for line in f:
            output = RequestAnalyzerBenchmarkOutput.from_json(line)
            outputs[output.input.id] = output
    return outputs


def write_outputs(outputs: list[RequestAnalyzerBenchmarkOutput], path: Path, mode: str = "a") -> None:
    with open(path, mode=mode) as f:
        for out in outputs:
            f.write(f"{out.to_json()}\n")


def get_candidate_drop_fields(spec: SpecType, prompt: str) -> list[str]:
    # Prioritize x and y channels and fields that appear in the prompt.
    # We do this because dropping a random gt field might not impact the plotting request.
    prompt_words = {word.strip().lower() for word in prompt.split()}

    def score_field(field: str, channel_name: str) -> float:
        score = 0.5
        if field.lower() in prompt_words:
            score += 1.0
        if channel_name.startswith("x") or channel_name.startswith("y"):
            score += 0.5
        return score

    fields: dict[str, float] = {}
    for encoding in get_spec_field(spec, "encoding"):  # encoding can be nested, e.g. "/spec/encoding"
        for channel_name, channel_def in encoding.items():
            if "field" not in channel_def:
                continue
            field = channel_def["field"]
            if isinstance(field, str):
                fields[field] = max(fields.get(field, 0.0), score_field(field, channel_name))
    return [field for field, _ in sorted(fields.items(), key=lambda x: x[1], reverse=True)]


def drop_ground_truth_columns(
    data: pd.DataFrame, ground_truth: SpecType | None, prompt: str, max_drop: int = -1
) -> tuple[pd.DataFrame, list[str] | None]:
    """Drop columns from data that appear in the ground truth spec."""
    if ground_truth is None:
        return data, None
    to_drop = []
    for field_name in get_candidate_drop_fields(ground_truth, prompt):
        if field_name not in data.columns:  # e.g. a new derived field
            continue
        to_drop.append(field_name)
    if max_drop >= 0:
        to_drop = to_drop[:max_drop]
    return data.drop(columns=to_drop), to_drop if len(to_drop) > 0 else None


def get_dropped_data(
    benchmark_type: RequestAnalyzerBenchmarkType, dataset_item: DatasetItem
) -> tuple[pd.DataFrame, list[str] | None]:
    prompt = "\n".join(dataset_item.prompt)
    dropped_fields = None
    match benchmark_type:
        case RequestAnalyzerBenchmarkType.NORMAL:
            data = dataset_item.data
        case RequestAnalyzerBenchmarkType.DROP_ONE_GROUND_TRUTH:
            data, dropped_fields = drop_ground_truth_columns(
                dataset_item.data, dataset_item.ground_truth, prompt, max_drop=1
            )
        case RequestAnalyzerBenchmarkType.DROP_ALL_GROUND_TRUTH:
            data, dropped_fields = drop_ground_truth_columns(
                dataset_item.data, dataset_item.ground_truth, prompt, max_drop=-1
            )
        case _ as unknown:
            assert_never(unknown)
    return data, dropped_fields


async def run_request_analyzer(
    config: RequestAnalyzerConfig,
    inp: RequestAnalyzerBenchmarkInput,
    dataset_item: DatasetItem,
) -> RequestAnalyzerBenchmarkOutput:
    data, dropped_fields = get_dropped_data(inp.benchmark_type, dataset_item)

    # N.B. We don't give the model the last spec.
    model = RequestAnalyzer.from_config(config, data)
    for i, prompt in enumerate(inp.prompt):
        await model.analyze_request(prompt, history_idx=i)
    response_history = model.get_response_history_full()

    return RequestAnalyzerBenchmarkOutput(
        input=inp,
        response_history=response_history,
        dropped_fields=dropped_fields,
    )


def compute_request_analyzer_benchmark_score(output: RequestAnalyzerBenchmarkOutput) -> float:
    warnings = []
    for prompt_response in output.response_history:
        if prompt_response.data_availability is not None:
            data_availability_response = prompt_response.data_availability[-1].response
            if data_availability_response is None:
                continue
            missing_data_warning, _ = get_data_availability_warning(data_availability_response)
            warnings.append(missing_data_warning)
        elif prompt_response.request_type is not None:
            request_type_response = prompt_response.request_type[-1].response
            if request_type_response is None:
                continue
            missing_data_warning, _ = get_request_type_missing_data_warning(request_type_response)
            warnings.append(missing_data_warning)

    match output.input.benchmark_type:
        case RequestAnalyzerBenchmarkType.NORMAL:
            # For "normal" requests, we expect the model to not warn. N.B. There may be false negatives.
            return 1.0 - sum(warnings) / len(warnings) if len(warnings) > 0 else float("nan")
        case RequestAnalyzerBenchmarkType.DROP_ONE_GROUND_TRUTH | RequestAnalyzerBenchmarkType.DROP_ALL_GROUND_TRUTH:
            if output.dropped_fields is None or len(output.dropped_fields) == 0:
                # if no fields are dropped, that's the same as "normal" but we already counted it
                return float("nan")
            # Without ground truth columns, we expect the model to warn of missing data.
            return sum(warnings) / len(warnings) if len(warnings) > 0 else float("nan")
        case _ as unknown:
            assert_never(unknown)


def compute_request_analyzer_benchmark_metrics(
    outputs: list[RequestAnalyzerBenchmarkOutput],
) -> dict[str, dict[str, float]]:
    all_scores = [compute_request_analyzer_benchmark_score(output) for output in outputs]
    all_ci = sample_mean_confidence_interval(all_scores)
    metrics = {"all": all_ci.to_dict("all")}
    metrics["all"]["count"] = len(all_scores)
    for benchmark_type in RequestAnalyzerBenchmarkType:
        scores = [
            compute_request_analyzer_benchmark_score(output)
            for output in outputs
            if output.input.benchmark_type == benchmark_type
        ]
        ci = sample_mean_confidence_interval(scores)
        metrics[benchmark_type.value] = ci.to_dict(benchmark_type.value)
        metrics[benchmark_type.value]["count"] = len(scores)
    return metrics


class BenchmarkDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        super().__init__(take_n=dataset._take_n, subset=dataset._ids_subset)
        self._dataset = dataset

    def all_ids(self) -> list[str]:
        ids = [
            f"{id_}/{benchmark_type}"
            for id_, benchmark_type in itertools.product(self._dataset.all_ids(), RequestAnalyzerBenchmarkType)
        ]
        return [id_ for id_ in ids if not self._is_invalid_id(id_)]

    def _is_invalid_id(self, id_: str) -> bool:
        item = self._parse_id(id_)
        assert item.metadata is not None
        benchmark_type = item.metadata["benchmark_type"]
        if benchmark_type != RequestAnalyzerBenchmarkType.NORMAL:
            # Exclude examples where nothing changes, or it's impossible to complete the task (empty df)
            new_df, dropped_fields = get_dropped_data(benchmark_type, item)
            if dropped_fields is None or len(dropped_fields) == 0 or new_df.shape[1] == 0:
                return True
        return False

    def _parse_id(self, id_: str) -> DatasetItem:
        dataset_item_id, benchmark_type = id_.rsplit("/", 1)
        dataset_item = self._dataset[dataset_item_id]
        dataset_item.id = id_
        metadata = dataset_item.metadata if dataset_item.metadata is not None else {}
        metadata["benchmark_type"] = RequestAnalyzerBenchmarkType(benchmark_type)
        dataset_item.metadata = metadata
        return dataset_item

    def __getitem__(self, id_: str) -> DatasetItem:
        return self._parse_id(id_)

    def get_config(self) -> dict[str, Any]:
        return self._dataset.get_config()


def run_request_analyzer_benchmark(
    dataset: Dataset,
    request_analyzer_config: RequestAnalyzerConfig,
    *,
    output_dir_or_file: Path,
    max_concurrent: int = 4,
    cli_args: dict[str, Any] | None = None,
) -> Path:
    wrapped_dataset = BenchmarkDatasetWrapper(dataset)

    run_config = RunConfig(
        dataset_config=wrapped_dataset.get_config(), request_analyzer_config=request_analyzer_config, cli_args=cli_args
    )
    if output_dir_or_file.is_file() or output_dir_or_file.suffix == ".jsonl":
        outputs_path = output_dir_or_file
    else:
        outputs_path = get_outputs_path(output_dir_or_file, run_config)
    config_path = get_run_config_path(outputs_path)
    run_config.dump_checked(config_path)

    logger.info(f"Saving outputs to: {outputs_path}")
    done_output_ids = set()
    if outputs_path.exists():
        for out_id in read_saved_outputs(outputs_path):
            done_output_ids.add(out_id)
    else:
        outputs_path.touch()
    todo_ids = []
    for id_ in wrapped_dataset.iter_ids():
        if id_ not in done_output_ids:
            todo_ids.append(id_)

    async def _aworker(id_: str) -> RequestAnalyzerBenchmarkOutput:
        dataset_item = wrapped_dataset[id_]
        assert dataset_item.metadata is not None
        benchmark_input = RequestAnalyzerBenchmarkInput(
            id=dataset_item.id,
            benchmark_type=dataset_item.metadata["benchmark_type"],
            prompt=dataset_item.prompt,
            ground_truth=dataset_item.ground_truth,
        )
        return await run_request_analyzer(request_analyzer_config, benchmark_input, dataset_item)

    asyncio.run(
        async_run_benchmark_items(
            todo_ids,
            _aworker,
            lambda out: write_outputs([out], outputs_path, mode="a"),
            max_concurrent=max_concurrent,
        )
    )

    return outputs_path


if __name__ == "__main__":
    from benchmark.datasets import NLVCorpusDataset

    dataset = NLVCorpusDataset(Path("dataset/nlv_corpus"), take_n=3)
    outputs_path = run_request_analyzer_benchmark(
        dataset, RequestAnalyzerConfig(), output_dir_or_file=Path("out/test.jsonl")
    )
    outputs = list(read_saved_outputs(outputs_path).values())
    print(compute_request_analyzer_benchmark_metrics(outputs))
