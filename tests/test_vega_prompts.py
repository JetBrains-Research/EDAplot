import commentjson
import pytest

from edaplot.prompt_utils import extract_tag_content
from edaplot.vega_chat.prompts import extract_model_response, sys_format_json_dict, sys_format_json_str


@pytest.mark.parametrize(
    ["s", "tag", "expected"],
    [
        ["<a></b>", "a", []],
        ["<a></a>", "a", [""]],
        ["</a>foo</a>", "a", []],
        ["</a>foo<a>", "a", []],
        ["<a>foo</a>", "a", ["foo"]],
        ["<a>foo1</a>,<a>foo2</a>", "a", ["foo1", "foo2"]],
        ["<a>foo1<a>,\n<a>foo2<a>", "a", ["foo1", "foo2"]],
        ["<a>foo1<a>\n,\n<a>foo2</a>", "a", ["foo1", "foo2"]],
        ["<a>foo1</a>\n,\n<a>foo2<a>", "a", ["foo1", "foo2"]],
    ],
)
def test_extract_tag_content(s: str, tag: str, expected: list[str]) -> None:
    got = extract_tag_content(s, tag)
    assert got == expected


@pytest.mark.parametrize(
    "specs",
    [
        [{"a": 1, "b": 2}],
        [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
    ],
)
def test_extract_model_response_jsons(specs: list[dict]) -> None:
    """Test for when the LLM generates multiple JSONs instead of one."""
    content = ",".join(commentjson.dumps(spec, indent=2) for spec in specs)
    content = sys_format_json_str(content)
    got_specs = extract_model_response(content).specs
    assert got_specs == specs

    content = "\n".join(sys_format_json_dict(spec) for spec in specs)
    got_specs = extract_model_response(content).specs
    assert got_specs == specs

    # LLM outputs array of specs [{},{},...]
    content = commentjson.dumps(specs, indent=2)
    content = sys_format_json_str(content)
    got_specs = extract_model_response(content).specs
    assert got_specs == specs

    for spec in specs:
        content = sys_format_json_dict(spec)
        got_specs = extract_model_response(content).specs
        assert got_specs == [spec]
