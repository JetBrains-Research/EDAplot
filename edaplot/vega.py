import copy
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Any, Literal, Self

import altair as alt
import pandas as pd
import vl_convert
from altair.utils.schemapi import SchemaValidationError
from jsonschema import ValidationError
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage

from edaplot.data_utils import df_is_datetime_column, spec_add_data, spec_remove_data
from edaplot.spec_utils import (
    SpecType,
    get_dict_value_by_path,
    get_scenegraph_field,
    get_spec_encoding_field_types,
    get_spec_field,
    get_spec_field_by_path,
    get_spec_views,
)
from edaplot.transform_utils import (
    get_transform_as_fields,
    get_transform_type,
    is_inline_transform_type,
    is_transform_type,
)
from edaplot.vega_chat.prompts import (
    VEGA_LITE_SCHEMA_URL,
    format_scenegraph_exception,
    format_schema_validation_error,
    get_empty_plot_correction_prompt,
    get_error_correction_prompt,
    get_manual_error_correction_prompt,
    get_multiple_jsons_error_prompt,
    get_transform_field_as_missing_prompt,
    get_transform_in_channel_error_prompt,
)

logger = getLogger(__name__)


class MessageType(StrEnum):
    USER = "user"
    """Actual user input"""
    SYSTEM = "system"
    """System prompt"""
    USER_ERROR_CORRECTION = "user_error_correction"
    """Prompt to fix errors"""
    AI_RESPONSE_VALID = "ai_response"
    """Valid AI response"""
    AI_RESPONSE_ERROR = "ai_response_error"
    """AI responded with an error"""

    @classmethod
    def is_ai_response(cls, message_type: Self) -> bool:
        return message_type in (cls.AI_RESPONSE_VALID, cls.AI_RESPONSE_ERROR)

    @classmethod
    def is_ai_response_error(cls, message_type: Self) -> bool:
        return message_type in (cls.AI_RESPONSE_ERROR,)

    @classmethod
    def create_message(cls, content: str, message_type: Self) -> BaseMessage:
        match message_type:
            case MessageType.USER:
                return HumanMessage(content)
            case MessageType.SYSTEM:
                return SystemMessage(content)
            case MessageType.USER_ERROR_CORRECTION:
                return HumanMessage(content)
            case _:
                raise ValueError(f"Cannot create message: {repr(message_type)}")


@dataclass(kw_only=True)
class SpecInfo:
    spec: SpecType
    is_spec_fixed: bool = False
    is_empty_chart: bool = True
    is_valid_schema: bool = False
    is_drawable: bool = False

    @classmethod
    def from_valid(cls, spec: SpecType) -> Self:
        return cls(spec=spec, is_spec_fixed=False, is_valid_schema=True, is_empty_chart=False, is_drawable=True)


@dataclass(kw_only=True)
class VegaMessage:
    # Container for Chat and Recommender messages
    message: BaseMessage
    message_type: MessageType
    spec_infos: list[SpecInfo]
    explanation: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        message_type = MessageType(d.pop("message_type"))
        spec_infos = [SpecInfo(**info) for info in d.pop("spec_infos")]
        return cls(**d, message_type=message_type, spec_infos=spec_infos)


def process_extracted_specs(specs: list[SpecType]) -> SpecType:
    if len(specs) > 1:
        # Retry request, since there are lots of concatenation edge cases to automatically fix this
        raise ValueError(get_multiple_jsons_error_prompt())
    else:
        spec = specs[0]

    # Don't delete nulls from the spec because sometimes setting a value to null is desired
    # [-] spec = spec_delete_nulls(spec)
    if "$schema" not in spec:
        spec["$schema"] = VEGA_LITE_SCHEMA_URL
    spec_remove_data(spec)
    return spec


def make_text_spec(content: str) -> SpecType:
    # Using altair to make sure we get a valid spec
    alt_chart = alt.Chart().mark_text(fontSize=20, text=content)
    spec = alt_chart.to_dict()
    spec_remove_data(spec)
    return spec


def to_altair_chart(
    spec: SpecType,
    df: pd.DataFrame,
) -> alt.Chart:
    spec_new = spec_add_data(spec, df)
    chart = alt.Chart.from_dict(spec_new)  # validate=False doesn't work with every spec even if valid...
    return chart


@dataclass(kw_only=True)
class SpecValidity:
    is_valid_schema: bool
    is_valid_scenegraph: bool
    is_empty_scenegraph: bool
    exc_scenegraph: ValueError | None = None
    exc_schema: ValidationError | None = None


def validate_spec(spec: SpecType, df: pd.DataFrame) -> SpecValidity:
    try:
        # - This can catch errors missed by schema validation, e.g. `Unrecognized function: sum`
        # - Data needs to be embedded into the json for vl_convert (=> can't use a url to a local csv)
        # - We could speed this up by subsampling df (but risking falsely empty charts?)
        # - TODO this sometimes prints errors but doesn't raise an exception?
        spec_with_data = spec_add_data(spec, df)
        scenegraph = vl_convert.vegalite_to_scenegraph(vl_spec=spec_with_data)
        is_valid_scenegraph = True
        # If no data is left after transformations, there is nothing to plot.
        # - The scenegraph approach is robust enough that we don't need is_transformed_data_empty.
        is_empty_scenegraph = is_chart_empty_scenegraph(scenegraph)
        exc_scenegraph = None
    except ValueError as e:
        logger.warning("scenegraph error", exc_info=e)
        is_empty_scenegraph = True
        is_valid_scenegraph = False
        exc_scenegraph = e

    try:
        # - Chart.from_dict has more fallbacks than Chart.validate
        # - We 'prefer' scenegraph checks because vega can render charts even if
        #   the schema validation fails.
        # - WARNING: trying to create a chart with lots of inline data takes forever!
        # - We are only using altair for schema validation, so it doesn't matter
        #   how much data we have as long as it looks valid. Down from minutes to milliseconds :)
        spec_with_data = spec_add_data(spec, df.head())
        alt.Chart.from_dict(spec_with_data)
        is_valid_schema = True
        exc_schema = None
    except ValidationError as e:
        logger.warning("schema validation error", exc_info=e)
        is_valid_schema = False
        exc_schema = e

    return SpecValidity(
        is_valid_schema=is_valid_schema,
        is_valid_scenegraph=is_valid_scenegraph,
        is_empty_scenegraph=is_empty_scenegraph,
        exc_scenegraph=exc_scenegraph,
        exc_schema=exc_schema,
    )


def is_transformed_data_empty(chart: alt.Chart) -> bool | None:
    try:
        # N.B. transformed_data works with inline transforms as well
        transformed_data = chart.transformed_data(row_limit=5)
        if transformed_data is None:
            return True
        elif isinstance(transformed_data, list):
            if all(len(d) == 0 for d in transformed_data):
                return True
        elif isinstance(transformed_data, pd.DataFrame):
            if len(transformed_data) == 0:
                return True
    except (ValueError, SchemaValidationError) as e:
        # vega fusion raises exceptions if the spec is "bad" and some transforms are not yet implemented
        logger.debug(e)
        pass
    return None


def is_chart_empty_scenegraph(scenegraph: dict) -> bool:
    # Conversion to scenegraph (usually) works even if the spec is not 100% valid
    # - https://github.com/vega/vega/tree/main/packages/vega-scenegraph
    # - https://vega.github.io/vega/docs/marks/
    # - A schema validation error does not imply an empty scenegraph.
    # - An empty scenegraph does not imply a schema validation error.
    def is_mark_group_empty(mark_role: dict[str, Any]) -> bool:
        if "items" not in mark_role or len(mark_role["items"]) == 0:
            return True
        # Certain marktypes have hidden items
        marktype = mark_role["marktype"]
        items = mark_role["items"]
        if marktype in ("line", "area", "trail"):
            # If all line segments are "hidden" and at least 2 points
            return len(items) <= 1 or all(not item.get("defined", True) for item in items)
        elif marktype in ("symbol",):
            # 0 size or undefined symbol
            return all(item.get("size", 42) == 0 or len(item.get("shape", "circle")) == 0 for item in items)
        elif marktype in ("shape",):
            # This is mostly used for geoshape
            return all(len(item.get("shape", "geoshape")) == 0 for item in items)
        elif marktype in ("rect",):
            # All bars have 0 width/height
            return all(item.get("height", 1) == 0 or item.get("width", 1) == 0 for item in items)
        return False

    mark_roles = get_scenegraph_field(scenegraph, "role", "mark")
    return all(is_mark_group_empty(mark_role) for mark_role in mark_roles)


@dataclass
class SpecFix:
    spec: SpecType
    spec_validity: SpecValidity | None
    reply: str | None


def append_reply(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None:
        return left
    return f"{left}\n{right}"


def spec_fix_merge(fst: SpecFix, snd: SpecFix) -> SpecFix:
    reply = append_reply(fst.reply, snd.reply)
    return SpecFix(snd.spec, snd.spec_validity, reply)


def validate_and_fix_spec(
    spec: SpecType,
    df: pd.DataFrame,
    *,
    retry_on_empty_plot: bool,
    max_reply_length: int,
    spec_history: list[SpecInfo] | None = None,
) -> SpecFix:
    # TODO Performance improvements: lazy spec validity and in-place spec modifications
    # We merge error correction replies from different sources to try and solve them all at once.
    spec_validity = validate_spec(spec, df)
    if not spec_validity.is_valid_scenegraph:
        spec_fix = spec_fix_faceting(spec, df, spec_validity)
    else:
        spec_fix = SpecFix(reply=None, spec=spec, spec_validity=spec_validity)

    if spec_fix is None:
        # Can't fix _invalid_ spec. Proceed with generic exception info.
        assert spec_validity.exc_scenegraph is not None
        reply = (
            format_schema_validation_error(spec_validity.exc_schema, max_length=max_reply_length)
            if spec_validity.exc_schema is not None
            else ""
        )
        reply += "\n" + format_scenegraph_exception(spec_validity.exc_scenegraph, max_length=max_reply_length)
        reply = get_error_correction_prompt(reply, max_length=max_reply_length)
        return SpecFix(reply=reply, spec=spec, spec_validity=spec_validity)

    if spec_history is not None and len(spec_history) > 0:
        assert spec_fix.spec_validity is not None
        fields_fix = spec_fix_restore_fields(spec, df, spec_fix.spec_validity, spec_history[-1])
        spec_fix = spec_fix_merge(spec_fix, fields_fix)

    # Try to fix transforms even if the spec is drawable (such errors are ignored when rendering)
    assert spec_fix.spec_validity is not None
    inline_transforms_fix = spec_fix_wrong_inline_transforms(spec_fix.spec, spec_fix.spec_validity)
    spec_fix = spec_fix_merge(spec_fix, inline_transforms_fix)
    assert spec_fix.spec_validity is not None
    transform_fix = spec_fix_transform_as_fields(spec_fix.spec, df, spec_fix.spec_validity)
    spec_fix = spec_fix_merge(spec_fix, transform_fix)

    # Fix incorrect date filtering transforms
    assert spec_fix.spec_validity is not None
    filter_fix = spec_fix_filter_dates(spec_fix.spec, df, spec_fix.spec_validity)
    spec_fix = spec_fix_merge(spec_fix, filter_fix)

    # Try to fix color issues (silently ignored)
    assert spec_fix.spec_validity is not None
    colors_fix = spec_fix_color_domain(spec_fix.spec, df, spec_fix.spec_validity)
    spec_fix = spec_fix_merge(spec_fix, colors_fix)

    assert spec_fix.spec_validity is not None
    null_domain_fix = spec_fix_null_scale_domain(spec_fix.spec, df, spec_fix.spec_validity)
    spec_fix = spec_fix_merge(spec_fix, null_domain_fix)

    # Remaining options:
    # a) already valid input spec (maybe empty)
    # b) a successful fix
    # c) a partial fix (drawable but empty plot)
    # d) not fixed, but we have a more detailed error message (reply)
    new_validity = spec_fix.spec_validity
    assert new_validity is not None
    if retry_on_empty_plot and new_validity.is_valid_scenegraph and new_validity.is_empty_scenegraph:
        spec_fix.reply = get_empty_plot_correction_prompt(spec_fix.spec, spec_fix.reply)
    return spec_fix


def spec_fix_faceting(spec: SpecType, df: pd.DataFrame, spec_validity: SpecValidity) -> SpecFix | None:
    """Attempt to fix incorrect faceting usage."""
    # Assumption: spec is not drawable and/or has an invalid schema
    assert not spec_validity.is_valid_scenegraph
    # See: https://vega.github.io/vega-lite/docs/facet.html
    # For now only single views (i.e., no hconcat...)
    if "encoding" in spec and "facet" in spec:
        # `encoding` and `facet` are not allowed to be in the same level so try inlining facet into `encoding`
        enc = spec["encoding"]
        is_enc_faceted = "row" in enc or "column" in enc or "facet" in enc
        if not is_enc_faceted:
            new_spec = copy.deepcopy(spec)
            facet = new_spec.pop("facet")
            new_spec["encoding"].update(facet)
            new_validity = validate_spec(new_spec, df)
            if new_validity.is_valid_scenegraph:
                return SpecFix(reply=None, spec=new_spec, spec_validity=new_validity)
        reply = get_manual_error_correction_prompt(
            "`encoding` and `facet` are not allowed to be in the same level. "
            "Inline the faceting into `encoding` using the `row` or `column` encoding channels."
        )
        return SpecFix(reply=reply, spec=spec, spec_validity=spec_validity)
    return None  # Can't fix and no clue what is wrong.


def spec_fix_transform_as_fields(spec: SpecType, df: pd.DataFrame, spec_validity: SpecValidity) -> SpecFix:
    """Finds missing "as" fields defined in transforms but not referenced in encodings."""
    # For now only single views (i.e., no hconcat...)
    reply: str | None = None

    # No transforms found in spec
    if "transform" not in spec:
        return SpecFix(spec, spec_validity, reply)
    transform = spec["transform"]
    if not isinstance(transform, list):
        reply = "The view-level `transform` must be an array."
        return SpecFix(spec, spec_validity, reply)
    assert isinstance(transform, list)

    # TODO check other transform steps (nested)
    # TODO ignore "as" fields that appear in other transforms
    encoding_fields = {}
    for encoding in get_spec_field(spec, "encoding"):  # encoding can be nested, e.g. "/spec/encoding"
        for channel_name, channel_def in encoding.items():
            if "field" not in channel_def:
                continue
            field = channel_def["field"]
            if isinstance(field, str):
                encoding_fields[field] = channel_name

    missing_as_fields = []
    for transform_as_field in get_transform_as_fields(transform):
        if transform_as_field is not None and transform_as_field not in encoding_fields:
            missing_as_fields.append(transform_as_field)
    reply = get_transform_field_as_missing_prompt(missing_as_fields) if len(missing_as_fields) > 0 else None
    return SpecFix(spec, spec_validity, reply)


def spec_fix_wrong_inline_transforms(spec: SpecType, spec_validity: SpecValidity) -> SpecFix:
    """Detect the problem of the LLM using view-level transforms in "encoding" as inline transforms."""
    bad_attributes = []
    for encoding in get_spec_field(spec, "encoding"):  # encoding can be nested, e.g. "/spec/encoding"
        for channel_name, channel_def in encoding.items():
            if not isinstance(channel_def, dict):
                continue
            for attribute in channel_def:
                is_bad = False
                if attribute == "transform":
                    is_bad = True
                elif attribute == "field":
                    field = channel_def["field"]
                    if isinstance(field, dict):
                        for field_dict_attr in field:
                            # The LLM sometimes places transforms inside `field`...
                            if field_dict_attr == "transform" or is_transform_type(attribute):
                                is_bad = True
                elif is_transform_type(attribute) and not is_inline_transform_type(attribute):
                    # e.g. "calculate" inside "y"
                    is_bad = True

                if is_bad:
                    bad_attributes.append((channel_name, attribute))

    reply: str | None = None
    if len(bad_attributes) > 0:
        reply = get_transform_in_channel_error_prompt(bad_attributes)
    return SpecFix(spec, spec_validity, reply)


def pd_timestamp_to_vl_datetime(ts: pd.Timestamp) -> dict[str, Any]:
    """Convert a pandas Timestamp to a Vega-Lite datetime object."""
    # https://vega.github.io/vega-lite/docs/datetime.html
    return {
        "year": ts.year,
        "month": ts.month,
        "date": ts.day,
        "hours": ts.hour,
        "minutes": ts.minute,
        "seconds": ts.second,
        "milliseconds": ts.microsecond // 1000,
    }


def _convert_to_vl_datetime(op_val: Any, filter_time_unit: Any) -> tuple[Any, str | None]:
    fixed_op_val = None
    reply = None
    if not isinstance(op_val, (str, int, bool, dict)):
        reply = f"Invalid type for temporal field '{op_val}' in 'filter'!"
    elif isinstance(op_val, int) or (isinstance(op_val, str) and op_val.isdigit()):
        # Assume nothing needs to be fixed if timeUnit is present
        if filter_time_unit is None:
            # Assume that a 4 digit number refers to a year and fix it automatically
            if len(str(op_val)) == 4:
                logger.debug(f"Converting '{op_val}' to a year datetime")
                fixed_op_val = {"year": op_val}
            else:
                # E.g. {"field": "date", "gte": 6} is not valid because we don't know what 6 refers to
                reply = f"'{op_val}' is not a valid Vega-Lite datetime! Are you missing a 'timeUnit'?"
            # Ignore if value is part of the "range" list
    elif isinstance(op_val, str):
        try:
            # op_val = "2010-06-01T00:00:00Z"
            ts = pd.to_datetime(op_val)
            vl_datetime = pd_timestamp_to_vl_datetime(ts)
            # If timeUnit is used together with a bad datetime:
            # e.g. {"timeUnit": "month", "gte": "1980-06-01"}, extract just the parsed timeUnit
            # to get {"timeUnit": "month", "gte": 6}
            if filter_time_unit is not None and filter_time_unit in vl_datetime:
                vl_datetime = vl_datetime[filter_time_unit]
            fixed_op_val = vl_datetime
        except (pd.errors.ParserError, ValueError) as e:
            logger.exception("fix_datetime_filter_predicate: pd.to_datetime failed")
            reply = f"Couldn't parse '{op_val}' as a valid Vega-Lite datetime!"
    return fixed_op_val, reply


def fix_datetime_filter_predicate(filt: dict[str, Any]) -> tuple[bool, str | None]:
    # Docs: https://vega.github.io/vega-lite/docs/predicate.html
    # Assume filt["field"] is a temporal field.
    # filt is modified in-place.
    vl_field_predicates = ("equal", "lt", "lte", "gt", "gte", "range", "oneOf")
    op = next((op for op in vl_field_predicates if op in filt), None)
    if op is None:
        return False, None  # Nothing to fix

    # Convert incorrect values to VL datetime expressions.
    reply = None
    is_fixed = False
    op_val = filt[op]
    filter_time_unit = filt.get("timeUnit")
    if op in ("range", "oneOf"):  # array based cases
        if op == "range" and (not isinstance(op_val, list) or not len(op_val) == 2):
            reply = '"range" in the filter transform must be a 2-element array!'
        elif op == "oneOf" and not isinstance(op_val, list):
            reply = '"oneOf" in the filter transform must be an array!'
        else:
            new_array = []
            array_changed = False
            for range_value in op_val:
                fixed_op_val, reply_ = _convert_to_vl_datetime(range_value, filter_time_unit)
                reply = append_reply(reply, reply_)
                if fixed_op_val is not None:
                    new_array.append(fixed_op_val)
                    array_changed = True
            if array_changed:
                filt[op] = new_array
                is_fixed = True
    else:
        fixed_op_val, reply = _convert_to_vl_datetime(op_val, filter_time_unit)
        if fixed_op_val is not None:
            filt[op] = fixed_op_val  # in-place!
            is_fixed = True
    return is_fixed, reply


def spec_is_temporal_field(spec: SpecType, df: pd.DataFrame, field: Any) -> bool:
    if not isinstance(field, str):
        return False
    # E.g., a calculated field is not in df.columns
    if field in df.columns and df_is_datetime_column(df, field):
        return True
    # Check if the LLM marked the field as "temporal"
    enc_field_types = get_spec_encoding_field_types(spec)
    return enc_field_types.get(field) == "temporal"


def validate_filter_transform_schema(filt_dict: dict[str, Any]) -> tuple[bool, str | None]:
    # Validate against the schema because there are too many edge cases to manually check
    try:
        alt.FilterTransform.from_dict(filt_dict)
        return True, None
    except ValidationError as err:
        reply = 'The "filter" transform does not conform to the Vega-Lite schema!'
        reply += "\n" + format_schema_validation_error(err, max_length=1024)
        return False, reply


def fix_filter_by_date(
    spec: SpecType, df: pd.DataFrame, filt_dict: dict[str, Any]
) -> tuple[bool, dict[str, Any], str | None]:
    """Fix incorrect datetime expressions in an otherwise _valid_ filter transform.

    For example, sometimes the LLM generates a datetime string instead of a valid VL datetime expression.
    { "field": "Year", "gte": "2010-06-01" } -> { "field": "Year", "gte": {"year": 2010, "month": 6, "date": 1} }
    """
    # Schema validation does not allow "range" values to be strings, but the LLM can generate datetime strings, which we
    # want to fix. So we patch the range check in that case.
    ranges = get_spec_field(filt_dict, "range")
    if len(ranges) > 0 and all(isinstance(r, list) and len(r) == 2 for r in ranges):
        patched_filt_dict = copy.deepcopy(filt_dict)
        for patched_range in get_spec_field(patched_filt_dict, "range"):
            patched_range[:] = [1, 2]  # modify in-place to valid schema
    else:
        patched_filt_dict = filt_dict
    valid_schema, reply = validate_filter_transform_schema(patched_filt_dict)
    if not valid_schema:
        return False, filt_dict, reply

    filt = filt_dict["filter"]
    if not isinstance(filt, dict):  # e.g. a string expression
        return False, filt_dict, None
    filt = copy.deepcopy(filt)  # make changes to a copy

    filts = []  # Multiple filters to fix for composition
    # https://vega.github.io/vega-lite/docs/predicate.html#composition
    composition_ops = ["and", "or", "not"]
    is_composition = False
    for op in composition_ops:
        if op in filt:
            op_filt = filt[op]
            if op in ("and", "or"):
                if not isinstance(op_filt, list):
                    return False, {}, 'The "and" section in the "filter" transform must be an array!'
                filts.extend(op_filt)
            else:
                filts.append(op_filt)
            break
    if not is_composition:
        filts.append(filt)

    fixed = False
    for flt in filts:
        if not isinstance(flt, dict):  # e.g., a string expression
            continue
        # Check if the field is a datetime object in the dataframe or in the spec, otherwise skip it
        if not spec_is_temporal_field(spec, df, flt.get("field")):
            continue
        is_flt_fixed, flt_reply = fix_datetime_filter_predicate(flt)
        reply = append_reply(reply, flt_reply)
        if is_flt_fixed:
            fixed = True

    if fixed:
        return True, {"filter": filt}, reply
    else:
        return False, filt_dict, reply


def spec_fix_filter_dates(spec: SpecType, df: pd.DataFrame, spec_validity: SpecValidity) -> SpecFix:
    """Fix incorrect filtering by dates in the "filter" transform."""
    reply = None
    spec_changed = False
    for transforms in get_spec_field(spec, "transform"):  # This handles vconcat etc. cases
        if not isinstance(transforms, list):
            continue
        for trans in transforms:
            if get_transform_type(trans, raise_error=False) != "filter":
                continue
            fixed, filter_dict, freply = fix_filter_by_date(spec, df, trans)
            reply = append_reply(reply, freply)
            if fixed:
                trans["filter"] = filter_dict["filter"]
                spec_changed = True

    if not spec_changed:
        return SpecFix(spec, spec_validity, reply)  # the reply could be not None (if invalid schema)
    else:
        new_validity = validate_spec(spec, df)
        return SpecFix(spec, new_validity, reply)


def spec_fix_restore_fields(
    spec: SpecType, df: pd.DataFrame, spec_validity: SpecValidity, prev_spec: SpecInfo
) -> SpecFix:
    """Add fields that were present in the previous spec but are missing in the current one."""

    def get_fields(s: SpecType) -> dict[tuple[str, ...], Any]:
        fields: dict[tuple[str, ...], Any] = {}
        for path, field in get_spec_field_by_path(s, ("encoding", "*", "field")):
            fields[path] = field  # can be a string or dict
        return fields

    # Find fields that used to exist but don't exist anymore
    old_fields = get_fields(prev_spec.spec)
    cur_fields = get_fields(spec)
    missing_fields = {}
    for old_path, old_field in old_fields.items():
        if old_path in cur_fields:  # field unchanged
            continue
        # Check if the old channel still exists
        old_channel_path = old_path[:-1]
        channel_exists, channel = get_dict_value_by_path(spec, old_channel_path)
        if not channel_exists:
            continue
        # The channel exists, but "field" is missing, so we'll add it if possible
        # "field" and "value" cannot coexist
        if "value" in channel:
            continue
        missing_fields[old_channel_path] = old_field

    if len(missing_fields) == 0:
        return SpecFix(spec, spec_validity, None)

    new_spec = copy.deepcopy(spec)
    for channel_path, new_field in missing_fields.items():
        logger.debug(f"Adding missing field {new_field} to channel {channel_path}")
        # Modify new spec by reference
        channel_exists, channel = get_dict_value_by_path(new_spec, channel_path)
        assert channel_exists and isinstance(channel, dict)
        channel["field"] = new_field
        # Also copy the "type" if necessary and available
        if "type" not in channel:
            old_type_path = channel_path + ("type",)
            old_type_exists, old_type_value = get_dict_value_by_path(prev_spec.spec, old_type_path)
            if old_type_exists:
                channel["type"] = old_type_value

    new_validity = validate_spec(new_spec, df)
    return SpecFix(new_spec, new_validity, None)


def spec_fix_null_scale_domain(spec: SpecType, df: pd.DataFrame, spec_validity: SpecValidity) -> SpecFix:
    """Remove "domain" from encoding channel "scale"s if it contains null values."""
    spec = copy.deepcopy(spec)
    for encoding in get_spec_field(spec, "encoding"):  # encoding can be nested, e.g. "/spec/encoding"
        for channel_name, channel_def in encoding.items():
            if "scale" not in channel_def:
                continue
            scale = channel_def["scale"]
            if "domain" not in scale:
                continue
            scale_domain = scale["domain"]
            if not isinstance(scale_domain, list):
                continue
            domain_has_null = any(x is None for x in scale_domain)
            if domain_has_null:
                scale.pop("domain")
    # ignore spec_validity
    return SpecFix(spec, spec_validity, None)


def spec_fix_color_domain(spec: SpecType, df: pd.DataFrame, spec_validity: SpecValidity) -> SpecFix:
    """Fix incorrect color domain usage in color/scale/domain."""
    new_spec = copy.deepcopy(spec)
    spec_changed = False
    for color_channel in get_spec_field(new_spec, "color"):
        if "scale" not in color_channel:
            continue
        color_scale = color_channel["scale"]
        if not isinstance(color_scale, dict):
            continue
        if "domain" not in color_scale:
            continue  # We fix only "domain" issues

        # See https://vega.github.io/vega-lite/docs/scale.html#domain
        color_field_type = color_channel.get("type")
        if color_field_type not in ("ordinal", "nominal"):
            continue
        # > domain for ordinal and nominal fields can be an array that lists valid input values
        color_domain = color_scale["domain"]
        if not isinstance(color_domain, list):
            continue

        possible_domain_values = None
        if "field" in color_channel:
            color_field = color_channel["field"]
            if isinstance(color_field, str) and color_field in df.columns:
                possible_domain_values = df[color_field].unique()

        # Sometimes the LLM generates nonsense values for the domain, resulting in colors to be ignored
        # If all values are good, keep them, otherwise remove the whole domain
        valid_domain_values = False
        if possible_domain_values is not None:
            valid_domain_values = set(color_domain).issubset(set(possible_domain_values))
        if not valid_domain_values:
            color_scale.pop("domain")
            spec_changed = True

    if spec_changed:
        new_validity = validate_spec(new_spec, df)
        return SpecFix(new_spec, new_validity, None)
    return SpecFix(spec, spec_validity, None)


AutoToolTip = Literal["data", "encoding", "none"]


def spec_set_auto_tooltip(spec: SpecType, tooltip_type: AutoToolTip, replace_tooltip: bool = False) -> SpecType:
    """Automatically set the tooltip.

    :param tooltip_type: 'data' to show all data columns, 'encoding' to show only columns in the "encoding" section,
    'none' to disable tooltips, 'ignore' to leave the spec as is.
    :param replace_tooltip: if True, the LLM generated tooltip will be replaced with the chosen option.
    """
    # Docs: https://vega.github.io/vega-lite/docs/tooltip.html
    spec = copy.deepcopy(spec)
    for view in get_spec_views(spec):
        if "mark" not in view:
            continue
        new_mark = copy.deepcopy(view["mark"])
        if isinstance(view["mark"], str):
            new_mark = {"type": view["mark"], "tooltip": None}
        match tooltip_type:
            case "data" | "encoding" as content_type:
                new_mark["tooltip"] = alt.TooltipContent(content_type).to_dict()
            case "none":
                new_mark["tooltip"] = None
        view["mark"] = new_mark

        # To force our tooltip, the "tooltip" channel must be removed (otherwise the "tooltip" channel has priority)
        if replace_tooltip:
            if "encoding" in spec and "tooltip" in spec["encoding"]:
                del spec["encoding"]["tooltip"]
    return spec


def spec_set_interactive_pan_zoom(spec: SpecType) -> SpecType:
    """Make the spec interactive by adding pan and zoom."""
    # https://vega.github.io/vega-lite/docs/bind.html#scale-binding
    spec = copy.deepcopy(spec)

    # Parameter names must be unique!
    all_param_names = set()
    for params in get_spec_field(spec, "params"):
        if not isinstance(params, list):
            continue
        for p in params:
            if isinstance(p, dict) and "name" in p:
                all_param_names.add(p["name"])

    def get_new_name() -> str:
        n = 0
        name = f"p{n}"
        while name in all_param_names:
            n += 1
            name = f"p{n}"
        all_param_names.add(name)
        return name

    for view in get_spec_views(spec):
        view_params = view.get("params", [])
        if not isinstance(view_params, list):
            continue
        is_interactive = any(
            isinstance(param, dict) and param.get("select") == "interval" and param.get("bind") == "scales"
            for param in view_params
        )
        if not is_interactive:
            view_params.append({"name": get_new_name(), "select": "interval", "bind": "scales"})
            view["params"] = view_params
    return spec


def spec_apply_effects(
    spec: SpecType,
    *,
    tooltip_type: AutoToolTip | None = "encoding",
    replace_tooltip: bool = False,
    interactive_pan_zoom: bool = True,
) -> SpecType:
    if tooltip_type is not None:
        spec = spec_set_auto_tooltip(spec, tooltip_type, replace_tooltip=replace_tooltip)
    if interactive_pan_zoom:
        spec = spec_set_interactive_pan_zoom(spec)
    return spec
