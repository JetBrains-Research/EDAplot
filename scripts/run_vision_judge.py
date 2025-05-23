import argparse
import pprint
import typing
from pathlib import Path
from typing import Any

from benchmark.benchmark_utils import init_benchmark_logging
from benchmark.metrics import compute_metrics
from benchmark.vega_chat_benchmark import read_saved_outputs, run_vision_judge_on_outputs
from benchmark.vision_judge import VisionJudgeConfig


def get_vision_judge_config_from_args(args: argparse.Namespace) -> VisionJudgeConfig:
    return VisionJudgeConfig(
        model_name=args.vision_judge_model_name,
        temperature=args.vision_judge_temperature,
        image_detail=args.vision_judge_image_detail,
        image_scale=args.vision_judge_image_scale,
        image_resize=args.vision_judge_image_resize,
        prompt_choice=args.vision_judge_prompt_choice,
    )


def add_vision_judge_config_args(parser: argparse.ArgumentParser) -> Any:
    conf = VisionJudgeConfig()
    g = parser.add_argument_group("Vision Judge Config")
    g.add_argument("--vision_judge_model_name", default=conf.model_name)
    g.add_argument("--vision_judge_temperature", default=conf.temperature)
    g.add_argument(
        "--vision_judge_image_detail",
        choices=typing.get_args(typing.get_type_hints(VisionJudgeConfig)["image_detail"]),
        default=conf.image_detail,
    )
    g.add_argument("--vision_judge_image_scale", type=float, default=conf.image_scale)
    g.add_argument("--vision_judge_image_resize", action=argparse.BooleanOptionalAction, default=conf.image_resize)
    g.add_argument(
        "--vision_judge_prompt_choice",
        choices=typing.get_args(typing.get_type_hints(VisionJudgeConfig)["prompt_choice"]),
        default=conf.prompt_choice,
    )
    return g


if __name__ == "__main__":
    init_benchmark_logging()

    def _main() -> None:
        parser = argparse.ArgumentParser()

        parser.add_argument("outputs_path", type=Path, help="Path to the .jsonl file of saved outputs.")
        add_vision_judge_config_args(parser)
        parser.add_argument(
            "--max_concurrent",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--overwrite", action="store_true", default=False, help="Overwrite existing vision judge results."
        )
        args = parser.parse_args()

        eval_config = get_vision_judge_config_from_args(args)

        # Add vision_judge_ prefix to not interfere with other benchmark cli args
        cli_args = vars(args)
        cli_args = {f"vision_judge_{k.removeprefix('vision_judge_')}": v for k, v in cli_args.items()}

        outputs_path = run_vision_judge_on_outputs(
            args.outputs_path,
            eval_config,
            max_concurrent=args.max_concurrent,
            cli_args=cli_args,
            overwrite=args.overwrite,
        )
        outputs_saved = read_saved_outputs(outputs_path)
        metrics = compute_metrics([out.to_metric_input() for out in outputs_saved])
        pprint.pprint(metrics)

    _main()
