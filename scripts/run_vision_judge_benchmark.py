import argparse
from pathlib import Path

from benchmark.benchmark_utils import init_benchmark_logging
from benchmark.vision_judge_benchmark import print_outputs, run_vision_judge_benchmark
from edaplot.paths import PATH_VISION_JUDGE_BENCHMARK_PATH
from scripts.run_vision_judge import add_vision_judge_config_args, get_vision_judge_config_from_args

if __name__ == "__main__":
    init_benchmark_logging()

    def _main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output_path",
            type=Path,
            default=Path("out/vision_judge_benchmark/"),
            help="Directory or .jsonl file to save outputs to.",
        )
        parser.add_argument(
            "--max_concurrent",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--input_path", type=Path, default=PATH_VISION_JUDGE_BENCHMARK_PATH, help="Path to the benchmark dataset."
        )
        add_vision_judge_config_args(parser)
        args = parser.parse_args()
        eval_config = get_vision_judge_config_from_args(args)

        outputs_path = run_vision_judge_benchmark(
            args.input_path,
            args.output_path,
            eval_config,
            max_concurrent=args.max_concurrent,
            cli_args=vars(args),
        )
        print_outputs(outputs_path)

    _main()
