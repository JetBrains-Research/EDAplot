from pathlib import Path

# N.B. consider making symlinks to the datasets
PATH_ROOT = Path(__file__).parent.parent
PATH_CHART_LLM = PATH_ROOT / "dataset" / "chart-llm"
PATH_NLV = PATH_ROOT / "dataset" / "nlv_corpus"
PATH_RESOURCES = PATH_ROOT / "tests" / "resources"
PATH_EVALS = PATH_RESOURCES / "evals"
PATH_VISION_JUDGE_BENCHMARK_DIR = PATH_RESOURCES / "vision_judge_benchmark"
PATH_VISION_JUDGE_BENCHMARK_PATH = PATH_VISION_JUDGE_BENCHMARK_DIR / "benchmark.yaml"
PATH_VISION_JUDGE_BENCHMARK_IMAGES = PATH_VISION_JUDGE_BENCHMARK_DIR / "images"
