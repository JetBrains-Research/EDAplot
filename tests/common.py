from benchmark.datasets import ChartLLMDataset, Dataset, NLVCorpusDataset, VegaDatasets
from edaplot.paths import PATH_CHART_LLM, PATH_NLV


def get_default_dataset(name: str) -> Dataset:
    if name == VegaDatasets.name():
        return VegaDatasets()
    if name == ChartLLMDataset.name():
        return ChartLLMDataset(PATH_CHART_LLM)
    if name == NLVCorpusDataset.name():
        return NLVCorpusDataset(PATH_NLV)
    raise ValueError(f"Unknown dataset: '{name}'")
