import abc
import asyncio
import dataclasses
import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Self, Sequence

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@dataclass(kw_only=True)
class BenchmarkRunConfig(abc.ABC):
    """Base class for benchmark run configs."""

    timestamp: str = dataclasses.field(default_factory=get_timestamp)

    @classmethod
    @abc.abstractmethod
    def from_path(cls, path: Path) -> Self:
        pass

    def dump(self, path: Path) -> None:
        def _json_default(obj: Any) -> Any:
            # For cli_args
            if isinstance(obj, Path):
                return str(obj)
            return obj

        j = json.dumps(dataclasses.asdict(self), indent=2, default=_json_default)
        path.write_text(j)
        logger.info(f"Wrote run config to: {path}")

    def dump_checked(self, config_path: Path) -> None:
        if config_path.exists():
            saved_config = BenchmarkRunConfig.from_path(config_path)
            if saved_config != self:
                logger.warning(f"Overwriting '{config_path}' ({saved_config}) with {self}")
        config_path.parent.mkdir(exist_ok=True, parents=True)
        self.dump(config_path)


def init_benchmark_logging(filename: str = "evaluate.log") -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        filename=filename,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )


def get_timestamp_path(output_dir: Path, iso_timestamp: str) -> Path:
    ts = datetime.datetime.fromisoformat(iso_timestamp)
    return output_dir / f"{ts.strftime('%Y%m%d_%H%M%S')}.jsonl"


def get_run_config_path(outputs_path: Path) -> Path:
    return outputs_path.with_suffix(".json")


async def async_run_benchmark_items[
    Inp, Out
](
    items: Sequence[Inp],
    aworker: Callable[[Inp], Awaitable[Out]],
    on_done: Callable[[Out], None],
    max_concurrent: int = 4,
) -> None:
    task_semaphore = asyncio.Semaphore(max_concurrent)  # Rate limit the number of concurrent chats

    async def worker(item: Inp) -> Out:
        async with task_semaphore:
            return await aworker(item)

    worker_tasks = (worker(id_) for id_ in items)
    for coro in tqdm(asyncio.as_completed(worker_tasks), total=len(items)):
        out = await coro
        on_done(out)
