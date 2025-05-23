import argparse
import pprint
import typing
from pathlib import Path

from benchmark.benchmark_utils import init_benchmark_logging
from benchmark.datasets import ChartLLMDataset, Dataset, NLVCorpusDataset, VegaDatasets
from benchmark.evals.eval_runner import print_eval_results, read_eval_inputs, run_evals
from benchmark.metrics import compute_metrics
from benchmark.models.base_models import EvalModelConfig, EvalModelType
from benchmark.models.coml4vis import CoMLConfig
from benchmark.models.lida import LIDAModelConfig
from benchmark.models.vega_chat import VegaChatEvalConfig
from benchmark.vega_chat_benchmark import (
    read_saved_outputs,
    run_dataset_benchmark,
    run_lida_self_eval_on_outputs,
    run_vision_judge_on_outputs,
)
from edaplot.data_prompts import DataDescriptionStrategy
from edaplot.paths import PATH_EVALS
from edaplot.vega_chat.vega_chat import ModelConfig
from scripts.run_lida_self_eval import add_lida_self_eval_config_args, get_lida_self_eval_config_from_args
from scripts.run_vision_judge import add_vision_judge_config_args, get_vision_judge_config_from_args


def get_dataset_from_args(args: argparse.Namespace) -> Dataset:
    dataset: Dataset
    if args.dataset == "vega_datasets":
        dataset = VegaDatasets(take_n=args.take_n)
    elif args.dataset == "chart_llm_gold":
        dataset = ChartLLMDataset(
            Path(args.dataset_dir),
            utterance_type=args.utterance_type,
            skip_invalid=args.skip_invalid,
            take_n=args.take_n,
        )
    elif args.dataset == "nlv_corpus":
        dataset = NLVCorpusDataset(
            Path(args.dataset_dir),
            sequential_only=args.sequential_only,
            sequential_outputs=args.multi_turn,
            single_turn_only=args.single_turn_only,
            visId_limit=args.visId,
            take_n=args.take_n,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return dataset


def get_model_config_from_args(args: argparse.Namespace) -> EvalModelConfig:
    match EvalModelType(args.model_type):
        case EvalModelType.VEGA_CHAT:
            return VegaChatEvalConfig(
                model_config=ModelConfig(
                    model_name=args.model_name,
                    temperature=args.temperature,
                    language=args.language,
                    n_ec_retries=args.n_ec_retries,
                    description_strategy=args.description_strategy,
                    message_trimmer_max_tokens=args.message_trimmer_max_tokens,
                    data_normalize_column_names=args.data_normalize_column_names,
                    data_parse_dates=args.data_parse_dates,
                ),
            )
        case EvalModelType.LIDA:
            return LIDAModelConfig(
                model_name=args.model_name,
                temperature=args.temperature,
                lida_summary=args.lida_summary,
                lida_library=args.lida_library,
                lida_fix_execute_errors=args.lida_fix_execute_errors,
            )
        case EvalModelType.CoML4Vis:
            return CoMLConfig(
                model_name=args.model_name,
                temperature=args.temperature,
            )
        case _ as unreachable:
            typing.assert_never(unreachable)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    default_eval_conf = VegaChatEvalConfig(model_config=ModelConfig())
    default_vega_chat_conf = default_eval_conf.model_config
    default_lida_conf = LIDAModelConfig()
    g = parser.add_argument_group("Model Config")
    g.add_argument("--model_type", default=default_eval_conf.model_type, choices=[e.value for e in EvalModelType])
    g.add_argument("--model_name", default=default_vega_chat_conf.model_name, type=str)
    g.add_argument("--temperature", default=default_vega_chat_conf.temperature, type=float)
    g.add_argument("--language", default=default_vega_chat_conf.language, nargs="?", const=None)
    g.add_argument(
        "--n_ec_retries",
        type=int,
        default=default_vega_chat_conf.n_ec_retries,
        help="Number of prompt error correction retries",
    )
    g.add_argument(
        "--description_strategy",
        default=default_vega_chat_conf.description_strategy,
        choices=typing.get_args(DataDescriptionStrategy),
    )
    g.add_argument("--message_trimmer_max_tokens", default=default_vega_chat_conf.message_trimmer_max_tokens, type=int)
    g.add_argument(
        "--data_normalize_column_names", default=default_vega_chat_conf.data_normalize_column_names, type=bool
    )
    g.add_argument("--data_parse_dates", default=default_vega_chat_conf.data_parse_dates, type=bool)

    g.add_argument("--lida_summary", default=default_lida_conf.lida_summary, type=str)
    g.add_argument("--lida_library", default=default_lida_conf.lida_library, type=str)
    g.add_argument(
        "--lida_fix_execute_errors",
        default=default_lida_conf.lida_fix_execute_errors,
        action=argparse.BooleanOptionalAction,
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Dataset")
    g.add_argument(
        "dataset", choices=["evals", "vega_datasets", "chart_llm_gold", "nlv_corpus"], help="Dataset to evaluate"
    )
    g.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the dataset to evaluate",
    )
    g.add_argument("--take_n", type=int, default=None, help="Take only the first n examples from the dataset")
    g.add_argument(
        "--utterance_type",
        choices=typing.get_args(ChartLLMDataset.UtteranceType),
        default="all",
        help="chart_llm_gold: limit to this utterance type.",
    )
    g.add_argument(
        "--skip_invalid", action="store_true", default=False, help="chart_llm_gold: skip examples with invalid schemas."
    )
    g.add_argument("--sequential_only", action="store_true", help="nlv_corpus: limit to `sequential=='y'` examples.")
    g.add_argument("--single_turn_only", action="store_true", help="nlv_corpus: limit to `sequential=='n'` prompts.")
    g.add_argument("--visId", type=str, default=None, help="nlv_corpus: limit to this visId.")
    g.add_argument(
        "--no_multi_turn",
        action="store_false",
        help="Evaluate multi-turn prompts in multiple turns (default: True). If False, treat multi-turn prompts as single prompts.",
        dest="multi_turn",
    )


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Benchmark")
    g.add_argument("--output_path", type=Path, default=Path("out"), help="Directory or .jsonl file to save outputs to.")
    g.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="If > 0, use this many cpu workers for running the benchmark. If == 0, run concurrently on the main thread. "
        "If < 0, use the number of cores available.",
    )
    g.add_argument(
        "--max_concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent requests to allow with asyncio (default: 4).",
    )


if __name__ == "__main__":
    init_benchmark_logging()

    def _main() -> None:
        parser = argparse.ArgumentParser()
        add_dataset_args(parser)
        add_model_args(parser)
        lida_self_eval_group = add_lida_self_eval_config_args(parser)
        lida_self_eval_group.add_argument(
            "--lida_self_eval", action=argparse.BooleanOptionalAction, default=False, help="Run LIDA self evaluation."
        )
        vision_judge_group = add_vision_judge_config_args(parser)
        vision_judge_group.add_argument(
            "--vision_judge", action=argparse.BooleanOptionalAction, default=False, help="Run Vision Judge evaluation."
        )
        add_benchmark_args(parser)
        args = parser.parse_args()

        model_config = get_model_config_from_args(args)

        if args.dataset == "evals":
            assert isinstance(model_config, VegaChatEvalConfig)
            evals_dir = args.dataset_dir or PATH_EVALS
            eval_inputs = read_eval_inputs(evals_dir)
            outputs_path = run_evals(
                eval_inputs,
                model_config.model_config,
                output_dir_or_file=args.output_path,
                max_concurrent=args.max_concurrent,
            )
            print_eval_results(outputs_path)
        else:
            dataset = get_dataset_from_args(args)
            outputs_path = run_dataset_benchmark(
                dataset,
                model_config,
                output_dir_or_file=args.output_path,
                max_workers=args.max_workers,
                max_concurrent=args.max_concurrent,
                cli_args=vars(args),
            )

            if args.lida_self_eval:
                lida_eval_config = get_lida_self_eval_config_from_args(args)
                outputs_path = run_lida_self_eval_on_outputs(
                    outputs_path,
                    lida_eval_config,
                    max_workers=max(args.max_workers, args.max_concurrent),
                    cli_args=vars(args),
                )

            if args.vision_judge:
                vision_judge_config = get_vision_judge_config_from_args(args)
                outputs_path = run_vision_judge_on_outputs(
                    outputs_path,
                    vision_judge_config,
                    max_concurrent=args.max_concurrent,
                    cli_args=vars(args),
                )

            outputs_saved = read_saved_outputs(outputs_path)
            metrics = compute_metrics([out.to_metric_input() for out in outputs_saved])
            pprint.pprint(metrics)

    _main()
