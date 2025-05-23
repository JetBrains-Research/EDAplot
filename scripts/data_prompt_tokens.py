# Script to determine how many columns we can include in the data prompt
import random
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from benchmark.datasets import VegaDatasets
from edaplot.data_prompts import DEFAULT_DATA_STRATEGY, DataDescriptionStrategy, get_data_description_prompt


def run(output_path: Path) -> None:
    rng = random.Random(0)
    dfs = [d.data.reset_index() for d in VegaDatasets()]

    def sample_column() -> tuple[str, pd.Series]:
        df = rng.choice(dfs)
        col = rng.choice(df.columns)
        return col, df[col]

    def sample_df(n_columns: int) -> pd.DataFrame:
        min_rows = min(len(df) for df in dfs)
        df = {}
        for i in range(n_columns):
            col_name, col_series = sample_column()
            while col_name in df:
                col_name = f"{col_name}_{i}"
            df[col_name] = col_series.sample(min_rows).reset_index(drop=True)
        return pd.DataFrame(df)

    repeat = 3
    cols_range = np.linspace(1, 2000, num=100)
    description_strategies: list[DataDescriptionStrategy] = ["head", "main"]
    model_names = ["gpt-4o-mini"]
    results: dict[str, list] = {
        "cols": [],
        "rows": [],
        "prompt": [],
        "tokens": [],
        "description_strategy": [],
        "model": [],
    }

    pbar = tqdm(total=len(model_names) * len(cols_range) * repeat * len(description_strategies))
    for model_name in model_names:
        model = ChatOpenAI(model=model_name)
        for n_cols in cols_range:
            for description_strategy in description_strategies:
                for _ in range(repeat):
                    df = sample_df(int(n_cols))
                    prompt = get_data_description_prompt(df, description_strategy)
                    results["cols"].append(n_cols)
                    results["rows"].append(len(df))
                    results["prompt"].append(len(prompt))
                    results["tokens"].append(model.get_num_tokens(prompt))
                    results["description_strategy"].append(description_strategy)
                    results["model"].append(model.model_name)
                    pbar.update(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


def analyze(output_path: Path) -> None:
    # Analyze the results by uploading the CSV to Vega Chat :)
    results_df = pd.read_csv(output_path)
    for context_size in [8192, 16384, 128000]:
        for perc in [0.25, 0.5, 0.75]:
            token_budget = perc * context_size
            mask = (results_df["tokens"] < token_budget) & (results_df["description_strategy"] == DEFAULT_DATA_STRATEGY)
            max_idx = results_df[mask]["tokens"].argmax()
            print(f"{context_size=} tokens | budget={token_budget} tokens ({perc*100}%)")
            print(results_df[mask].iloc[max_idx].to_string())
            print()

    for model_name in results_df["model"].unique():
        for description_strategy in results_df["description_strategy"].unique():
            mask = (results_df["description_strategy"] == description_strategy) & (results_df["model"] == model_name)
            tokens_per_col = results_df[mask].apply(lambda r: r["tokens"] / r["cols"], axis=1).mean()
            print(f"[{model_name}, {description_strategy}] mean tokens per column: {tokens_per_col:.2f}")


if __name__ == "__main__":
    output_path = Path("out") / "data_prompt_threshold.csv"
    if not output_path.exists():
        run(output_path)
    analyze(output_path)
