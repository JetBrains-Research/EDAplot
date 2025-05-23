# EDAplot (VegaChat)

This repository contains a snapshot of the code used for the paper "Generating and Evaluating Declarative Charts Using Large Language Models".

## Usage

Run the interactive Streamlit prototype locally with:
```bash
poetry run python -m streamlit run frontend/app.py
```

To use the code as a library, look into [api.py](./edaplot/api.py).

## Evaluation

### Setup

Download evaluation datasets:
- [NLV Corpus](dataset/nlv_corpus/README.md) is included
- [chart-llm](https://github.com/hyungkwonko/chart-llm) should be cloned into `./dataset/`

### Benchmarks

Example for running the NLV Corpus benchmark:
```bash
poetry run python -m scripts.run_benchmark nlv_corpus --dataset_dir dataset/nlv_corpus --output_path out/benchmarks
```

Run the interactive results report with:
```bash
poetry run python -m streamlit run benchmark/reports/vega_chat_benchmark_report.py out/benchmarks
```
where `out` is the path to the directory containing the saved outputs.

### Evals

Our set of custom test cases ([_evals_](tests/resources/evals)) are defined as `yaml` files.
Each eval specifies the _actions_ to take and the _checks_ to perform after each action.

Run the evals with:
```bash
poetry run python -m scripts.run_benchmark evals --output_path out/evals
```

Run the interactive results report with:
```bash
poetry run python -m streamlit run benchmark/reports/evals_report.py out/evals
```
where `out` is the path to the directory containing the saved outputs.

Update existing results with new checks using:
```bash
poetry run python -m scripts.run_eval_checks out/evals/
```

### Request Analyzer

Run the request analyzer benchmark with:
```bash
poetry run python -m scripts.run_request_analyzer_benchmark --dataset_dir dataset/chart-llm --take_n 180 --output_path out/request_analyzer_benchmark/ chart_llm_gold
```

View the results with:
```bash
poetry run python -m streamlit run benchmark/reports/request_analyzer_benchmark_report.py out/request_analyzer_benchmark/
```


### LLM as a judge

#### Vision Judge

The vision judge uses a multimodal LLM to compare the generated image to the reference image.
It can be used to compare results from different plotting libraries (e.g., matplotlib and Vega-Lite).

To run the vision judge evaluation on existing outputs use:
```bash
poetry run python -m scripts.run_vision_judge example.jsonl
```
or use the `--vision_judge` flag together with `scripts/run_benchmark.py`

##### Vision Judge Benchmark
To evaluate the vision judge, we use a separate [benchmark](tests/resources/vision_judge_benchmark).

Run it with:
```bash
poetry run python -m scripts.run_vision_judge_benchmark
```

View the results with:
```bash
poetry run python -m streamlit run benchmark/reports/vision_judge_benchmark_report.py out/vision_judge_benchmark/
```

#### LIDA Self-Evaluation

[LIDA](https://github.com/microsoft/lida)'s self-evaluation can be run with:
```bash
poetry run python -m scripts.run_lida_self_eval example.jsonl
```

## Configuring dev environment

1. [Install poetry](https://python-poetry.org/docs/#installing-with-the-official-installer): `poetry self update 2.1.3`
2. Install dependencies:
```bash
poetry sync --no-root
```
3. Run `poetry run pre-commit install`
4. Add LLM providers' keys to env variables

Run tests with:
```bash
poetry run pytest tests
```
For some tests you need to first download the [Evaluation datasets](#evaluation).


### Docker

Build the image and run the container:
```bash
docker build -f frontend.Dockerfile -t edaplot .
docker run --rm -p 8501:8501 -e OPENAI_API_KEY -t edaplot
```
