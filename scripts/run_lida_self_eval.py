import argparse
import pprint
from pathlib import Path
from typing import Any

from benchmark.benchmark_utils import init_benchmark_logging
from benchmark.lida_self_eval import LIDASelfEvalConfig
from benchmark.metrics import compute_metrics
from benchmark.vega_chat_benchmark import read_saved_outputs, run_lida_self_eval_on_outputs


def get_lida_self_eval_config_from_args(args: argparse.Namespace) -> LIDASelfEvalConfig:
    return LIDASelfEvalConfig(
        eval_model_name=args.lida_eval_model_name,
        eval_temperature=args.lida_eval_temperature,
        lida_eval_choice=args.lida_eval_choice,
    )


def add_lida_self_eval_config_args(parser: argparse.ArgumentParser) -> Any:
    conf = LIDASelfEvalConfig()
    g = parser.add_argument_group("LIDA Self-Eval Config")
    g.add_argument("--lida_eval_model_name", default=conf.eval_model_name)
    g.add_argument("--lida_eval_temperature", default=conf.eval_temperature)
    g.add_argument(
        "--lida_eval_choice",
        default=conf.lida_eval_choice,
        choices=["code", "spec"],
        help="Use 'spec' to judge the Vega-Lite spec and 'code' to judge the generated code (for LIDA). "
        "If 'spec', 'lida_library' will be ignored.",
    )
    return g


if __name__ == "__main__":
    init_benchmark_logging()

    def _main() -> None:
        parser = argparse.ArgumentParser()

        parser.add_argument("outputs_path", type=Path, help="Path to the .jsonl file of saved outputs.")
        add_lida_self_eval_config_args(parser)
        parser.add_argument(
            "--max_workers",
            type=int,
            default=4,
            help="Number of threads. 0 for single threaded.",
        )
        args = parser.parse_args()

        eval_config = get_lida_self_eval_config_from_args(args)

        # Add lida_eval_ prefix to not interfere with other benchmark cli args
        cli_args = vars(args)
        cli_args = {f"lida_eval_{k.removeprefix('lida_eval_')}": v for k, v in cli_args.items()}

        outputs_path = run_lida_self_eval_on_outputs(
            args.outputs_path,
            eval_config,
            max_workers=args.max_workers,
            cli_args=cli_args,
        )
        outputs_saved = read_saved_outputs(outputs_path)
        metrics = compute_metrics([out.to_metric_input() for out in outputs_saved])
        pprint.pprint(metrics)

    _main()
