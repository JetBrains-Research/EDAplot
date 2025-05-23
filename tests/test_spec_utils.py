from typing import Any

import pytest

from edaplot.spec_utils import (
    ListIndex,
    get_dict_value_by_path,
    get_scenegraph_field,
    get_spec_color_ranges,
    get_spec_field,
    get_spec_field_by_path,
    get_spec_keys,
    get_spec_leaf_key_values,
    get_spec_marks,
    get_spec_paths,
    get_spec_transform_paths,
)
from edaplot.vega import SpecType


@pytest.mark.parametrize(
    ["inp", "out"],
    [
        ({}, [()]),
        ([], [()]),
        ({"a": {}, "b": []}, [("a",), ("b",)]),
        ({"a": "va", "b": ["b0"]}, [("a", "va"), ("b", ListIndex(0), "b0")]),
        ({"a": {"b": {"c": "d"}}}, [("a", "b", "c", "d")]),
        (["a", "b"], [(ListIndex(0), "a"), (ListIndex(1), "b")]),
    ],
)
def test_get_spec_paths(inp: Any, out: list[tuple]) -> None:
    got = get_spec_paths(inp)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "out"],
    [
        ({}, []),
        ([], []),
        ({"a": {}, "b": ["vb"]}, ["a", "b"]),
        ({"a": {}, "b": [{"a": 0}]}, ["a", "b", "a"]),
        ({"a": "va", "b": ["b0"]}, ["a", "b"]),
        ({"a": {"b": {"c": "d"}}}, ["a", "b", "c"]),
        (["a", "b"], []),
    ],
)
def test_get_spec_keys(inp: Any, out: list[str]) -> None:
    got = get_spec_keys(inp)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "out"],
    [
        ({}, []),
        ({"a": "va", "b": ["b0"]}, [("a", "va"), ("b", ListIndex(0), "b0")]),
        ({"b": [{"a": 0}]}, [("a", 0)]),
        ({"a": {}, "b": ["vb"]}, [("b", ListIndex(0), "vb")]),
        ({"a": {"b": {"c": "d"}}}, [("c", "d")]),
    ],
)
def test_get_spec_key_value_paths(inp: SpecType, out: list[tuple]) -> None:
    got = get_spec_leaf_key_values(inp)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "field", "out"],
    [
        ({}, "a", []),
        ({"a": "va", "b": ["b0"]}, "a", ["va"]),
        ({"b": [{"a": {"c": 0}}]}, "a", [{"c": 0}]),
        ({"a": {"a": {"c": "d"}}, "b": {"a": {}}}, "a", [{"a": {"c": "d"}}, {}]),
    ],
)
def test_get_spec_field(inp: SpecType, field: str, out: list[Any]) -> None:
    got = get_spec_field(inp, field)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "field", "value", "out"],
    [
        ({}, "a", None, []),
        ({"a": "va", "b": ["b0"]}, "a", None, [{"a": "va", "b": ["b0"]}]),
        ({"b": [{"a": {"c": 0}}]}, "a", None, [{"a": {"c": 0}}]),
        ({"a": {"b": {"b": "d"}}, "c": {"b": {}}}, "b", None, [{"b": {"b": "d"}}, {"b": {}}]),
        ({"a": {"b": {"b": "d"}}, "c": {"b": {}}}, "b", "d", [{"b": "d"}]),
    ],
)
def test_get_scenegraph_field(inp: SpecType, field: str, value: Any, out: list[tuple]) -> None:
    got = get_scenegraph_field(inp, field, value)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "out"],
    [
        ({}, []),
        ({"mark": "bar", "foo": ["mark"]}, ["bar"]),
        ({"mark": "bar", "foo": [{"mark": {"type": "line"}}]}, ["bar", "line"]),
    ],
)
def test_get_spec_marks(inp: SpecType, out: list[str]) -> None:
    got = get_spec_marks(inp)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "out"],
    [
        ({}, []),
        ({"transform": "va", "b": ["b0"]}, [("va",)]),
        ({"transform": []}, []),
        ({"transform": [{"a": 0}]}, [(ListIndex(0), "a", 0)]),
        ({"a": {"b": {"transform": [{"bk": "bv"}]}, "c": {"transform": "cv"}}}, [(ListIndex(0), "bk", "bv"), ("cv",)]),
    ],
)
def test_get_spec_transforms(inp: SpecType, out: list[tuple]) -> None:
    got = get_spec_transform_paths(inp)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "path", "out"],
    [
        ({}, ("a",), []),
        ({"a": 1}, ("a",), [(("a",), 1)]),
        ({"b": {"a": 1}}, ("a",), []),
        (
            {"b": {"a": 1}},
            (
                "*",
                "a",
            ),
            [(("b", "a"), 1)],
        ),
        (
            {"b": {"a": 1}},
            (
                "b",
                "*",
            ),
            [(("b", "a"), 1)],
        ),
        ({"b": {"a": 1}}, ("*",), [(("b",), {"a": 1})]),
        ({"b": [{"a": {"c": 0}}]}, ("a",), []),  # lists not supported
    ],
)
def test_get_spec_field_by_path(inp: SpecType, path: tuple[str, ...], out: list[tuple[tuple[str], Any]]) -> None:
    got = get_spec_field_by_path(inp, path)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "path", "out"],
    [
        ({}, ("a",), (False, None)),
        ({"a": 1}, (), (False, None)),
        ({"a": 1}, ("a",), (True, 1)),
        ({"b": {"a": 1}}, ("a",), (False, None)),
        (
            {"b": {"a": 1}},
            (
                "b",
                "a",
            ),
            (True, 1),
        ),
        ({"b": [{"a": {"c": 0}}]}, ("b",), (True, [{"a": {"c": 0}}])),
        ({"b": [{"a": {"c": 0}}]}, ("b", "a"), (False, None)),
    ],
)
def test_get_dict_value_by_path(inp: SpecType, path: tuple[str, ...], out: tuple[bool, Any]) -> None:
    got = get_dict_value_by_path(inp, path)
    assert got == out


@pytest.mark.parametrize(
    ["inp", "out"],
    [
        ({"encoding": {"color": {"field": "foo", "scale": {"range": ["red", "blue"]}}}}, [["red", "blue"]]),
        ({"encoding": {"color": {"field": "foo", "scale": {"range": "viridis"}}}}, [["viridis"]]),
        ({"encoding": {"x": {"field": "foo", "scale": {"range": ["red", "blue"]}}}}, [["red", "blue"]]),
        (
            {
                "encoding": {
                    "x": {"field": "foo", "scale": {"range": ["x1", "x2"]}},
                    "color": {"field": "foo", "scale": {"range": ["c1", "c2"]}},
                }
            },
            [["x1", "x2"], ["c1", "c2"]],
        ),
    ],
)
def test_get_spec_color_ranges(inp: SpecType, out: list[list[str]]) -> None:
    got = get_spec_color_ranges(inp)
    assert got == out
