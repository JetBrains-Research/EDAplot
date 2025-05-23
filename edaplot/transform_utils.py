from typing import Any

import altair as alt
from jsonschema import ValidationError

from edaplot.spec_utils import get_spec_has_encoding_field
from edaplot.vega_chat.prompts import format_schema_validation_error

ALT_TRANSFORMS: dict[str, type[alt.Transform]] = {
    "aggregate": alt.AggregateTransform,
    "bin": alt.BinTransform,
    "calculate": alt.CalculateTransform,
    "density": alt.DensityTransform,
    "extent": alt.ExtentTransform,
    "filter": alt.FilterTransform,
    "flatten": alt.FlattenTransform,
    "fold": alt.FoldTransform,
    "impute": alt.ImputeTransform,
    "joinaggregate": alt.JoinAggregateTransform,
    "loess": alt.LoessTransform,
    "lookup": alt.LookupTransform,
    "pivot": alt.PivotTransform,
    "quantile": alt.QuantileTransform,
    "regression": alt.RegressionTransform,
    "sample": alt.SampleTransform,
    "stack": alt.StackTransform,
    "timeUnit": alt.TimeUnitTransform,
    "window": alt.WindowTransform,
}


def get_transform_type(transform: dict[str, Any], *, raise_error: bool = True) -> str:
    for k in transform:
        if k in ALT_TRANSFORMS:
            return k
    if raise_error:
        raise ValueError(f"Unknown transform type: {transform}!")
    return "null"


def alt_validate_transform_schema(transform: dict[str, Any]) -> str | None:
    transform_type = get_transform_type(transform)
    alt_cls = ALT_TRANSFORMS[transform_type]
    try:
        alt_cls.from_dict(transform)
        return None
    except ValidationError as err:
        reply = f'The "{transform_type}" transform does not conform to the Vega-Lite schema!'
        reply += "\n" + format_schema_validation_error(err, max_length=1024)
        return reply


def get_transform_as_fields(transforms: list[dict[str, Any]]) -> list[str | None]:
    as_fields: list[str | None] = []
    for trans in transforms:
        as_field = trans.get("as")
        if as_field is None or not isinstance(as_field, str):
            as_fields.append(None)
        else:
            as_fields.append(as_field)
    return as_fields


def is_valid_transform(
    spec: dict[str, Any], transform: dict[str, Any], validate_transform: bool = True
) -> tuple[bool, str | None]:
    # Docs: https://vega.github.io/vega-lite/docs/transform.html
    if validate_transform:
        schema_reply = alt_validate_transform_schema(transform)
        if schema_reply is not None:
            return False, schema_reply

    if "calculate" in transform:
        if "as" not in transform:
            return False, 'Missing required "as" field in "calculate" transform!'
        as_field = transform["as"]
        if not isinstance(as_field, str):
            return False, 'The "as" field in the "calculate" transform must be a string!'
        if not get_spec_has_encoding_field(spec, as_field):
            return (
                False,
                f'The field "{as_field}" defined in the "calculate" transform is not used in any of the encoding channels!',
            )
    elif "filter" in transform:
        filter_value = transform["filter"]
        if isinstance(filter_value, dict):
            if "field" in filter_value:
                field = filter_value["field"]
                if not isinstance(field, str):
                    return False, 'The "field" field in the "filter" transform must be a string!'
                if not get_spec_has_encoding_field(spec, field):
                    # TODO does it make sense to filter a field not visualized?
                    return (
                        False,
                        f'The field "{field}" defined in the "filter" transform is not used in any of the encoding channels!',
                    )

    # TODO more checks
    return True, None


def is_inline_transform_type(transform_type: str) -> bool:
    return transform_type in ("bin", "timeUnit", "aggregate", "sort", "stack")


def is_transform_type(field: str) -> bool:
    return field in ALT_TRANSFORMS
