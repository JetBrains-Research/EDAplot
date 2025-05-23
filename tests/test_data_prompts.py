import pytest

from benchmark.datasets import Dataset
from edaplot.data_prompts import DataDescriptionStrategy, get_data_description_prompt
from edaplot.data_utils import df_preprocess
from tests.test_datasets import default_dataset


@pytest.mark.parametrize(
    "description_strategy",
    ["head", "main", pytest.param("lida", marks=pytest.mark.xfail(reason="LIDA fails with dates"))],
)
def test_data_description_on_dataset(default_dataset: Dataset, description_strategy: DataDescriptionStrategy) -> None:
    for item in default_dataset:
        df = df_preprocess(item.data, normalize_column_names=False, parse_dates=True)
        prompt = get_data_description_prompt(df, description_strategy=description_strategy)
        assert len(prompt) > 0
