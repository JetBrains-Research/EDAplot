from pathlib import Path

import edaplot.paths


def test_paths_exist() -> None:
    for var, value in vars(edaplot.paths).items():
        if var.startswith("PATH_"):
            assert isinstance(value, Path)
            assert value.exists(), f"{value} doesn't exist!"
