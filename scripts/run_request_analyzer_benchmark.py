import argparse
import pprint

from benchmark.benchmark_utils import init_benchmark_logging
from benchmark.request_analyzer_benchmark import (
    compute_request_analyzer_benchmark_metrics,
    read_saved_outputs,
    run_request_analyzer_benchmark,
)
from edaplot.request_analyzer.request_analyzer import RequestAnalyzerConfig
from scripts.run_benchmark import add_benchmark_args, add_dataset_args, get_dataset_from_args

if __name__ == "__main__":
    init_benchmark_logging()

    def _main() -> None:
        parser = argparse.ArgumentParser()
        add_dataset_args(parser)
        add_benchmark_args(parser)
        args = parser.parse_args()

        dataset = get_dataset_from_args(args)
        model_config = RequestAnalyzerConfig()
        outputs_path = run_request_analyzer_benchmark(
            dataset,
            model_config,
            output_dir_or_file=args.output_path,
            max_concurrent=args.max_concurrent,
            cli_args=vars(args),
        )
        outputs_saved = read_saved_outputs(outputs_path)
        metrics = compute_request_analyzer_benchmark_metrics(list(outputs_saved.values()))
        pprint.pprint(metrics)

    _main()
