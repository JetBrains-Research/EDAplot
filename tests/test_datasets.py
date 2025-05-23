from typing import Any

import pandas as pd
import pytest

from benchmark.datasets import (
    ChartLLMDataset,
    Dataset,
    DatasetItem,
    NLVCorpusDataset,
    VegaDatasets,
    load_dataset_from_config,
)
from edaplot.data_utils import spec_remove_data
from edaplot.paths import PATH_CHART_LLM, PATH_NLV
from edaplot.spec_utils import SpecType, get_spec_marks
from edaplot.vega import validate_spec


@pytest.fixture(
    params=[
        (VegaDatasets, {}),
        (ChartLLMDataset, dict(repo_path=PATH_CHART_LLM)),
        (NLVCorpusDataset, dict(dataset_path=PATH_NLV)),
    ],
    ids=[VegaDatasets.name(), ChartLLMDataset.name(), NLVCorpusDataset.name()],
)
def default_dataset(request: Any) -> Dataset:
    dataset_cls, params = request.param
    return dataset_cls(**params)


def _assert_validity(spec: SpecType | None, df: pd.DataFrame, check_schema: bool = True) -> None:
    assert spec is not None
    validity = validate_spec(spec, df)
    assert validity.is_valid_scenegraph
    if check_schema:
        assert validity.is_valid_schema
    if "geoshape" not in get_spec_marks(spec):
        # TODO fix geoshape issues
        assert not validity.is_empty_scenegraph


def _assert_prompt_and_data(item: DatasetItem) -> None:
    assert len(item.prompt) > 0
    for prompt in item.prompt:
        assert len(prompt) > 0
    assert len(item.data) > 0


def test_vega_datasets() -> None:
    dataset = VegaDatasets()
    assert len(dataset) > 0
    for item in dataset:
        _assert_prompt_and_data(item)


@pytest.mark.parametrize(["skip_invalid"], [[True], [False]])
def test_chart_llm_dataset(skip_invalid: bool) -> None:
    dataset = ChartLLMDataset(repo_path=PATH_CHART_LLM, utterance_type="command", skip_invalid=skip_invalid)
    assert len(dataset) > 0
    for item in dataset:
        assert item.metadata is not None
        if item.id.startswith("2_"):
            # This example has external data which does not exist anymore, so remove those references.
            spec_remove_data(item.ground_truth)
        _assert_prompt_and_data(item)
        _assert_validity(item.ground_truth, item.data, check_schema=dataset._skip_invalid)
        png_path = dataset.get_png_path(item.id)
        assert png_path.exists(), f"{png_path} doesn't exist!"


def test_chart_llm_dataset_invalid() -> None:
    """Make sure all marked invalid ids are actually (still) invalid."""
    dataset = ChartLLMDataset(repo_path=PATH_CHART_LLM, utterance_type="command", skip_invalid=False)
    assert len(dataset) > 0
    for item in dataset:
        idx = int(item.id.split("_")[0])
        if idx in dataset.INVALID_INDICES:
            assert item.ground_truth is not None
            validity = validate_spec(item.ground_truth, item.data)
            assert not validity.is_valid_schema


def test_nlv_dataset() -> None:
    # Specs repeat multiple times for different prompts and it would take too long to verify each prompt-spec pair.
    dataset = NLVCorpusDataset(PATH_NLV)
    assert len(dataset) > 0
    seen_specs = set()
    for item in dataset:
        _assert_prompt_and_data(item)
        assert item.metadata is not None
        key = f"{item.metadata['dataset']}-{item.metadata['visId']}"
        if key in seen_specs:
            continue
        seen_specs.add(key)
        _assert_validity(item.ground_truth, item.data)


def test_load_dataset_from_config(default_dataset: Dataset) -> None:
    config = default_dataset.get_config()
    loaded_dataset = load_dataset_from_config(config)
    assert len(loaded_dataset) == len(default_dataset)
