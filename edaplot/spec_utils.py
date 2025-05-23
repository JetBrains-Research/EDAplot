import typing
from typing import Any, TypeAlias

ListIndex = typing.NamedTuple("ListIndex", [("i", int)])

SpecType: TypeAlias = dict[str, Any]


def get_spec_paths(d: Any) -> list[tuple]:
    out: list[tuple] = []
    if isinstance(d, dict):
        if len(d) == 0:
            out.append(tuple())
        for k, v in d.items():
            out.extend((k, *x) for x in get_spec_paths(v))
    elif isinstance(d, list):
        if len(d) == 0:
            out.append(tuple())
        for i, v in enumerate(d):
            out.extend((ListIndex(i), *x) for x in get_spec_paths(v))
    else:
        out.append((d,))
    return out


def spec_paths_ignore_list_order(paths: list[tuple]) -> list[tuple]:
    new_paths = []
    for path in paths:
        new_path = tuple(ListIndex(0) if isinstance(p, ListIndex) else p for p in path)
        new_paths.append(new_path)
    return new_paths


def get_spec_keys(d: Any) -> list[str]:
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out.append(k)
            out.extend(get_spec_keys(v))
    elif isinstance(d, list):
        for v in d:
            out.extend(get_spec_keys(v))
    return out


def get_spec_leaf_key_values(spec: SpecType) -> list[tuple]:
    """Return key-value pairs of items in the spec.

    Example:
    ```
    {
      "mark": "line",
      "encoding": {
        "x": {
          "field": "date"
        }
      }
    }
    ```
    will return: `[("mark", "line"), ("field", "date")]`.
    """
    paths = get_spec_paths(spec)
    paths = [ListIndex(0) if isinstance(p, ListIndex) else p for p in paths]
    kvs = []
    for path in paths:
        i = len(path) - 2
        while i >= 0 and isinstance(path[i], ListIndex):
            i -= 1
        if i >= 0 and not isinstance(path[i], ListIndex):
            kvs.append(path[i:])
    return kvs


def get_spec_field(d: Any, field: str) -> list[Any]:
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            if k == field:
                out.append(v)
            else:
                out.extend(get_spec_field(v, field))
    elif isinstance(d, list):
        for v in d:
            out.extend(get_spec_field(v, field))
    return out


def get_spec_marks(d: SpecType) -> list[str]:
    """Returns `["bar"]` if input is `{"mark": {"type": "bar"}}`.

    A chart can have multiple "mark"s (e.g. vconcat).
    "mark" is not always a direct child of a spec, it can be deeply nested (e.g. in vconcat, hconcat, ...).
    """
    marks = []
    for mark in get_spec_field(d, "mark"):
        if isinstance(mark, str):
            marks.append(mark)
        elif isinstance(mark, dict):
            if "type" in mark:
                marks.append(mark["type"])
    return marks


def get_spec_transform_paths(d: SpecType) -> list[tuple]:
    paths = []
    for trans in get_spec_field(d, "transform"):
        # ignore `transform: []`
        tps = get_spec_paths(trans)
        tps = [p for p in tps if len(p) > 0 and not isinstance(p[-1], ListIndex)]
        paths += tps
    return paths


def spec_delete_nulls(d: Any) -> Any:
    if isinstance(d, dict):
        remove_keys = []
        for k in d.keys():
            if d[k] is None:
                remove_keys.append(k)
            else:
                d[k] = spec_delete_nulls(d[k])
        for k in remove_keys:
            del d[k]
        return d
    elif isinstance(d, list):
        return [spec_delete_nulls(e) for e in d]
    else:
        return d


def get_dict_value_by_path(d: SpecType, path: tuple[str, ...]) -> tuple[bool, Any]:
    """Get a nested dict value by its direct path."""
    if len(path) == 0 or not isinstance(d, dict):
        return False, None
    k = path[0]
    if k in d:
        if len(path) == 1:
            return True, d[k]
        return get_dict_value_by_path(d[k], path[1:])
    return False, None


def get_spec_field_by_path(d: Any, path: tuple[str, ...]) -> list[tuple[tuple[str], Any]]:
    """Get all the fields that match the given path. Use '*' for a wildcard key."""
    if len(path) == 0 or not isinstance(d, dict):
        return []
    found = []
    for k, v in d.items():
        if k == path[0] or path[0] == "*":
            if len(path) == 1:
                found.append(((k,), v))
            else:
                next_found = get_spec_field_by_path(v, path[1:])
                found.extend(((k,) + next_p, v) for (next_p, v) in next_found)  # type: ignore
    return found


def get_spec_color_ranges(d: SpecType) -> list[list[str]]:
    # https://vega.github.io/vega-lite/docs/scale.html#2-setting-the-range-property-to-an-array-of-valid-css-color-strings
    # N.B. There exist other ways to specify colors! (e.g. config-level)
    found = []
    for path, range_ in get_spec_field_by_path(d, ("encoding", "*", "scale", "range")):
        if isinstance(range_, list):
            found.append(range_)
        elif isinstance(range_, str):
            found.append([range_])
    for path, range_ in get_spec_field_by_path(d, ("encoding", "color", "value")):
        if isinstance(range_, str):
            found.append([range_])
    for mark in get_spec_field(d, "mark"):
        if isinstance(mark, dict) and isinstance(mark.get("color"), str):
            found.append([mark["color"]])
    return found


def get_spec_encoding_field_types(spec: SpecType) -> dict[str, str | None]:
    """Return a mapping from "field" to "type" for every encoding field."""
    fields = {}
    for encoding in get_spec_field(spec, "encoding"):  # encoding can be nested, e.g. "/spec/encoding"
        for channel_name, channel_def in encoding.items():
            if "field" not in channel_def:
                continue
            field = channel_def["field"]
            field_type = channel_def.get("type")
            if isinstance(field, str):
                field_type = None if not isinstance(field_type, str) else field_type
                fields[field] = field_type
    return fields


def get_spec_has_encoding_field(spec: SpecType, field: str) -> bool:
    spec_fields = get_spec_encoding_field_types(spec)
    return field in spec_fields


def get_spec_views(spec: SpecType) -> typing.Iterable[SpecType]:
    for k in ("hconcat", "vconcat", "concat", "layer"):
        if k in spec and isinstance(spec[k], list):
            for view in spec[k]:
                yield from get_spec_views(view)
    if "spec" in spec:  # for layer and facet
        yield spec["spec"]
    yield spec


def spec_concatenate_views(specs: list[SpecType], op: typing.Literal["hconcat", "vconcat"] = "hconcat") -> SpecType:
    # See: https://vega.github.io/vega-lite/docs/concat.html
    # TODO extract common view parts ($schema, data, transform) into the top-level
    return {op: specs}


def get_scenegraph_field(d: Any, key: str, value: Any = None) -> list[dict]:
    # The scenegraph is just a dict
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key and value is None or v == value:
                out.append(d)
            else:
                out.extend(get_scenegraph_field(v, key, value))
    elif isinstance(d, list):
        for v in d:
            out.extend(get_scenegraph_field(v, key, value))
    return out


def spec_is_empty(spec: SpecType) -> bool:
    if len(spec) == 0:
        return True
    if len(spec) == 1 and "$schema" in spec:
        return True
    return False
