import argparse
import concurrent.futures
import logging
from concurrent.futures import as_completed
from pathlib import Path

from tqdm import tqdm

from benchmark.datasets import Dataset
from benchmark.vega_chat_benchmark import (
    SavedOutput,
    get_saved_output_paths,
    load_dataset_from_outputs_path,
    read_saved_outputs,
    write_outputs,
)
from edaplot.vega import MessageType, validate_spec

logger = logging.getLogger(__name__)


def update_saved_validity(output: SavedOutput, dataset: Dataset) -> SavedOutput:
    data = dataset[output.id]
    output.ground_truth = data.ground_truth  # Useful when making changes to the ground truths
    for msg in output.messages:
        if MessageType.is_ai_response(msg.message_type):
            for spec_info in msg.spec_infos:
                spec_validity = validate_spec(spec_info.spec, data.data)
                spec_info.is_valid_schema = spec_validity.is_valid_schema
                spec_info.is_empty_chart = spec_validity.is_empty_scenegraph
                spec_info.is_drawable = spec_validity.is_valid_scenegraph
    return output


def update_saved_outputs(
    outputs: list[SavedOutput], dataset: Dataset, *, max_workers: int | None = 0
) -> list[SavedOutput]:
    new_outputs = []
    if max_workers == 0:
        for new_output in tqdm(outputs):
            new_outputs.append(update_saved_validity(new_output, dataset))
    else:
        max_workers = None if max_workers is None or max_workers < 0 else max_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(update_saved_validity, output, dataset) for output in outputs]
            for future in tqdm(as_completed(futures), total=len(futures)):
                new_output = future.result()
                new_outputs.append(new_output)
    return new_outputs


if __name__ == "__main__":

    def _main() -> None:
        parser = argparse.ArgumentParser(description="Update saved parsed messages (e.g. plot spec, explanation...)")
        parser.add_argument("outputs_path", type=Path, help="Saved outputs path")
        parser.add_argument(
            "--overwrite", action="store_true", help="Overwrite existing files (default: don't overwrite)"
        )
        parser.add_argument("--num_workers", type=int, default=0, help="Number of workers to use.")
        args = parser.parse_args()

        paths = get_saved_output_paths(args.outputs_path)
        for path in (pbar := tqdm(paths)):
            pbar.set_description(str(path))
            dataset = load_dataset_from_outputs_path(path)
            outputs = read_saved_outputs(path)
            new_outputs = update_saved_outputs(outputs, dataset, max_workers=args.num_workers)
            new_path = path if args.overwrite else path.with_stem(path.stem + "_updated")
            write_outputs(new_outputs, new_path, mode="w")

    _main()
