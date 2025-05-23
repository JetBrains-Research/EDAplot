import io
import json

import pandas as pd
import pytest
from commentjson import commentjson
from PIL import Image

from edaplot.image_utils import vl_to_png_bytes
from edaplot.spec_utils import SpecType

iris_values_json = """
[
  {"sepalLength": 5.1, "sepalWidth": 3.5, "petalLength": 1.4, "petalWidth": 0.2, "species": "setosa"},
  {"sepalLength": 4.9, "sepalWidth": 3.0, "petalLength": 1.4, "petalWidth": 0.2, "species": "setosa"},
  {"sepalLength": 4.7, "sepalWidth": 3.2, "petalLength": 1.3, "petalWidth": 0.2, "species": "setosa"},
  {"sepalLength": 4.6, "sepalWidth": 3.1, "petalLength": 1.5, "petalWidth": 0.2, "species": "setosa"},
  {"sepalLength": 5.0, "sepalWidth": 3.6, "petalLength": 1.4, "petalWidth": 0.2, "species": "setosa"},
  {"sepalLength": 5.4, "sepalWidth": 3.9, "petalLength": 1.7, "petalWidth": 0.4, "species": "setosa"},
  {"sepalLength": 4.6, "sepalWidth": 3.4, "petalLength": 1.4, "petalWidth": 0.3, "species": "setosa"}
]
"""


@pytest.fixture
def test_data() -> tuple[pd.DataFrame, SpecType]:
    data = json.loads(iris_values_json)
    df = pd.json_normalize(data)
    json_part = """
    {"mark":"bar","encoding":{"x":{"field":"sepalLength","bin":true,"title":"Sepal Length"},"y":{"aggregate":"count","title":"Count"}},"width":400,"height":300}
    """
    config = commentjson.loads(json_part)
    return df, config


def test_altair_save_to_png(test_data: tuple[pd.DataFrame, SpecType]) -> None:
    df, config = test_data
    png = vl_to_png_bytes(config, df)
    # Check if the png variable is not None
    assert png is not None, "PNG data should not be None"
    # Check if the png variable is a bytes object
    assert isinstance(png, bytes), "PNG data should be of type bytes"
    # Verify the PNG data by attempting to open it with PIL
    try:
        image = Image.open(io.BytesIO(png))
        image.verify()
    except (IOError, SyntaxError) as e:
        raise AssertionError("Invalid PNG data generated") from e
