import base64
import io
import logging
from pathlib import Path

import pandas as pd
import PIL.Image
import vl_convert
from PIL import Image

from edaplot.data_utils import spec_add_data
from edaplot.spec_utils import SpecType

logger = logging.getLogger(__name__)


def encode_image_bytes_to_base64(img: bytes) -> str:
    # Format according to https://platform.openai.com/docs/guides/vision
    # N.B. LIDA decodes to "ascii", which should be equivalent.
    return base64.b64encode(img).decode("utf-8")


def decode_image_base64(b64: str) -> bytes:
    return base64.b64decode(b64)


def encode_local_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return encode_image_bytes_to_base64(image_file.read())


def vl_to_png_bytes(vl_spec: SpecType, df: pd.DataFrame, scale: float = 1.0) -> bytes | None:
    spec = spec_add_data(vl_spec, df)
    try:
        png_data = vl_convert.vegalite_to_png(vl_spec=spec, scale=scale)
    except ValueError as e:
        logger.exception("Vega-Lite to PNG conversion failed")
        png_data = None

    # Check if the image is too large to be useful (probably)
    if png_data is not None:
        try:
            Image.open(io.BytesIO(png_data))
        except PIL.Image.DecompressionBombError as e:
            logger.exception("PNG data is too large!")
            png_data = None
    return png_data


def vl_to_png_base64(spec: SpecType, df: pd.DataFrame, scale: float = 1.0) -> str | None:
    """Render a Vega-Lite spec to a base64 encoded PNG image."""
    png_bytes = vl_to_png_bytes(spec, df, scale=scale)
    if png_bytes is None:
        return None
    return encode_image_bytes_to_base64(png_bytes)


def get_image_dimensions(img: bytes) -> tuple[int, int]:
    img_pil = Image.open(io.BytesIO(img))
    return img_pil.size


def get_square_resize_factor(wh_src: tuple[int, int], wh_target: tuple[int, int]) -> float:
    """Resize factor to make the source image fit into the square of the target image."""
    src_a = max(wh_src)
    target_a = max(wh_target)
    return target_a / src_a


def resize_png_image(img: bytes, wh_target: tuple[int, int]) -> bytes:
    """Resize an image to the given width and height."""
    img_pil = Image.open(io.BytesIO(img))
    img_resized = img_pil.resize(wh_target)
    img_resized_bytes = io.BytesIO()
    img_resized.save(img_resized_bytes, format="PNG")
    return img_resized_bytes.getvalue()


def resize_png_image_to_square(img: bytes, wh_target: tuple[int, int]) -> bytes:
    """Resize a PNG image to fit the square of the target image."""
    wh_src = get_image_dimensions(img)
    resize_factor = get_square_resize_factor(wh_src, wh_target)
    if resize_factor == 1.0:
        return img
    return resize_png_image(img, (int(wh_src[0] * resize_factor), int(wh_src[1] * resize_factor)))
