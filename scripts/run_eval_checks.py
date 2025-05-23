import argparse
from pathlib import Path

from benchmark.evals.eval_runner import update_all_eval_checks

if __name__ == "__main__":

    def _main() -> None:
        parser = argparse.ArgumentParser(description="Update saved parsed messages (e.g. plot spec, explanation...)")
        parser.add_argument("outputs_path", type=Path, help="Saved outputs path")
        args = parser.parse_args()
        update_all_eval_checks(args.outputs_path)

    _main()
