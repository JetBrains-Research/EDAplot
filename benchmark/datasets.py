import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Self

import pandas as pd
import vega_datasets

from edaplot.vega import SpecType

logger = logging.getLogger(__name__)


@dataclass
class DatasetItem:
    id: str
    prompt: list[str]  # list for multi-turn prompts
    data: pd.DataFrame
    ground_truth: SpecType | None = None
    metadata: dict[str, Any] | None = None


class Dataset(ABC):
    def __init__(self, *, take_n: int | None = None, subset: list[str] | None = None):
        self._take_n = take_n
        self._ids_subset = subset if subset is not None else None

    @abstractmethod
    def all_ids(self) -> list[str]:
        """Return a list of all available example ids. Use this to perform custom filtering."""
        pass

    def iter_ids(self) -> list[str]:
        ids = self._ids_subset if self._ids_subset is not None else self.all_ids()
        if self._take_n is not None:
            ids = ids[: self._take_n]
        return ids

    @abstractmethod
    def __getitem__(self, id_: str) -> DatasetItem:
        pass

    def __len__(self) -> int:
        return len(self.iter_ids())

    def __iter__(self) -> Iterator[DatasetItem]:
        for id_ in self.iter_ids():
            yield self[id_]

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        config = config.copy()
        name = config.pop("name")
        assert name == cls.name()
        return cls(**config)


class VegaDatasets(Dataset):
    def __init__(self, *, take_n: int | None = None, subset: list[str] | None = None):
        super().__init__(take_n=take_n, subset=subset)
        self._datasets = vega_datasets.local_data.list_datasets()

    def all_ids(self) -> list[str]:
        return self._datasets

    def __getitem__(self, id_: str) -> DatasetItem:
        assert id_ in self._datasets
        df = vega_datasets.data(id_)
        prompt = ["Plot the data"]
        return DatasetItem(id=id_, prompt=prompt, data=df, ground_truth=None)

    @classmethod
    def name(cls) -> str:
        return "vega_datasets"


class ChartLLMDataset(Dataset):
    UtteranceType = Literal["command", "query", "question", "all"]
    INVALID_INDICES = {2, 5, 6, 13, 14, 15, 16, 17, 18, 20, 28, 32, 35, 37, 39, 40, 41, 46, 47}

    def __init__(
        self,
        repo_path: Path,
        utterance_type: UtteranceType = "all",
        skip_invalid: bool = False,
        take_n: int | None = None,
        subset: list[str] | None = None,
    ):
        super().__init__(take_n=take_n, subset=subset)
        self._repo_path = repo_path
        self._utterance_type = utterance_type
        self._skip_invalid = skip_invalid

        self._gold_csv_path = self._repo_path / "exp/gold/result/gold.csv"
        self._gold_vegas_dir = self._repo_path / "docs/data/chart_48_in"
        self._gold_csvs_dir = self._repo_path / "docs/data/csv_48_process"
        self._gold_png_dir = self._repo_path / "docs/data/chart_48_img"
        self._gold_df = pd.read_csv(self._gold_csv_path)

    def all_ids(self) -> list[str]:
        # These vega lite specs produce schema ValidationErrors, but they work in the online vega editor
        #  a) skip them
        #  b) don't use altair for validation
        valid_indices = set(range(len(self._gold_df)))
        if self._skip_invalid:
            valid_indices.difference_update(self.INVALID_INDICES)
        valid_indices.difference_update({37})  # TODO fix this example

        utt_types = ["command", "query", "question"] if self._utterance_type == "all" else [self._utterance_type]
        ids = []
        for i in sorted(valid_indices):
            for utt in utt_types:
                ids.append(f"{i}_{utt}")
        return ids

    def get_png_path(self, id_: str) -> Path:
        idx, _ = id_.split("_")
        idx = int(idx)
        gold_row = self._gold_df.iloc[idx]
        level = gold_row["level"]
        level_int = 0
        if level == "simple":
            level_int = 1
        elif level == "medium":
            level_int = 2
        elif level == "complex":
            level_int = 3
        elif level == "extracomplex":
            level_int = 4
        return self._gold_png_dir / f"{level_int}.{level}" / f"visualization ({idx}).png"

    def __getitem__(self, id_: str) -> DatasetItem:
        idx, utt_field = id_.split("_")
        idx = int(idx)
        csv_path = self._gold_csvs_dir / f"d_{idx:02}.csv"
        df = pd.read_csv(csv_path)
        vega_path = self._gold_vegas_dir / f"vl_{idx:02}.vl.json"
        vega_gt = json.loads(vega_path.read_text())
        gold_row = self._gold_df.iloc[idx]
        prompt = [str(gold_row[utt_field])]  # TODO multi-turn mode for \n prompts
        metadata = {"utterance": utt_field, "difficulty": gold_row["level"]}
        return DatasetItem(id=id_, prompt=prompt, data=df, ground_truth=vega_gt, metadata=metadata)

    @classmethod
    def name(cls) -> str:
        return "chart_llm_gold"

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(repo_path=str(self._repo_path), utterance_type=self._utterance_type)
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        config = config.copy()
        name = config.pop("name")
        assert name == cls.name()
        config["repo_path"] = Path(config["repo_path"])
        return cls(**config)


class NLVCorpusDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        *,
        sequential_only: bool = False,
        sequential_outputs: bool = True,
        single_turn_only: bool = False,
        visId_limit: str | None = None,
        take_n: int | None = None,
        subset: list[str] | None = None,
    ):
        """NLV Corpus Dataset.

        :param sequential_only: Limit to `sequential == 'y'` outputs.
        :param single_turn_only: Limit to single-turn examples.
        :param sequential_outputs: Prepare sequential outputs for multi-turn evaluation (default).
            If False, multi-turn prompts are concatenated into a single prompt.
        :param visId_limit: Limit to examples with the specified visId.
        """
        super().__init__(take_n=take_n, subset=subset)
        self._dataset_path = dataset_path
        self._sequential_only = sequential_only
        self._sequential_outputs = sequential_outputs
        self._single_turn_only = single_turn_only
        self._visId_limit = visId_limit

        self._vl_specs = json.loads((self._dataset_path / "vlSpecs.json").read_text())
        self._dfs = {  # There are some utf8 encoding errors so ignore them
            p.stem: pd.read_csv(p, encoding_errors="ignore") for p in (self._dataset_path / "datasets").glob("*.csv")
        }
        self._corpus = pd.read_csv(self._dataset_path / "NLV_Corpus.csv")

    def all_ids(self) -> list[str]:
        ids = []
        for i in range(len(self._corpus)):
            row = self._corpus.iloc[i]
            if self._visId_limit is not None and row["visId"] != self._visId_limit:
                continue
            if self._sequential_only and row["sequential"] != "y":
                continue
            if self._single_turn_only and row["sequential"] != "n":
                continue
            ids.append(str(i))
        return ids

    def __getitem__(self, id_: str) -> DatasetItem:
        idx = int(id_)
        row = self._corpus.iloc[idx]

        if row["sequential"] == "y":
            # Sequential prompts are separated by |
            prompt = [p.strip() for p in row["Utterance Set"].split("|")]
        else:
            prompt = [row["Utterance Set"]]
        if not self._sequential_outputs:
            prompt = ["\n".join(prompt)]

        vis_type = row["visId"]
        dataset = row["dataset"].lower()
        spec_id = f"{dataset}-{vis_type}"
        ground_truth = self._vl_specs[spec_id]
        metadata = {"dataset": dataset, "visId": vis_type, "sequential": row["sequential"]}
        return DatasetItem(id=id_, prompt=prompt, data=self._dfs[dataset], ground_truth=ground_truth, metadata=metadata)

    @classmethod
    def name(cls) -> str:
        return "nlv_corpus"

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            dataset_path=str(self._dataset_path),
            sequential_only=self._sequential_only,
            sequential_outputs=self._sequential_outputs,
            single_turn_only=self._single_turn_only,
            visId_limit=self._visId_limit,
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        config = config.copy()
        name = config.pop("name")
        assert name == cls.name()
        config["dataset_path"] = Path(config["dataset_path"])
        return cls(**config)


def load_dataset_from_config(config: dict[str, Any]) -> Dataset:
    name = config["name"]
    if name == VegaDatasets.name():
        return VegaDatasets.from_config(config)
    if name == ChartLLMDataset.name():
        return ChartLLMDataset.from_config(config)
    if name == NLVCorpusDataset.name():
        return NLVCorpusDataset.from_config(config)
    raise ValueError(f"Unknown dataset: '{name}'")
