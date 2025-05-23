import math

import pytest

from benchmark.vision_judge import (
    MatPlotAgentResponse,
    parse_matplotagent_prompt_response,
    parse_plotting_benchmark_prompt_response,
)


@pytest.mark.parametrize(
    "content, expected",
    [
        ("[FINAL SCORE]: 85", 85.0),  # Valid score
        ("[FINAL SCORE]: 100", 100.0),
        ("[FINAL SCORE]: 1000", 1000.0),
        ("[FINAL SCORE]: 50.0", 50.0),
        ("[FINAL SCORE]: 42.69", 42.69),
        ("[final score]: 42", 42.0),  # Lower case score label
        ("[FINAL SCORE]: abc", float("nan")),  # Malformed score
        ("This text does not contain a score.", float("nan")),  # No score
        ("Some intro text... [FINAL SCORE]: 99 ... Some closing text.", 99.0),  # Extra text
        ("\t [FINAL SCORE]:    73\n", 73.0),  # Different formatting and spacing
    ],
)
def test_parse_plotting_benchmark_prompt_response(content: str, expected: float) -> None:
    got = parse_plotting_benchmark_prompt_response(content)
    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == expected


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"score": 85, "label": "valid"}', (85.0, "valid")),  # Valid input
        ('{"score": 100.5, "label": "excellent"}', (100.5, "excellent")),  # Valid float score
        ('{"score": "abc", "label": "invalid"}', (float("nan"), "invalid")),  # Non-numeric score
        ('{"label": "missing score"}', (float("nan"), "missing score")),  # Missing score
        ('{"score": 75}', (75.0, None)),  # Missing label
        ('{"score": 60, "label": 80}', (60.0, "80")),  # Non-string label
        ('[{"score": 95, "label": "array"}]', (float("nan"), None)),  # Invalid JSON structure
        ('{"score": 42.5, "label": "half"}', (42.5, "half")),  # Decimal score
        ("", (float("nan"), None)),  # Empty content
        ("This is not a JSON.", (float("nan"), None)),  # Invalid JSON format
    ],
)
def test_parse_matplotagent_prompt_response(content: str, expected: tuple[float, str]) -> None:
    expected = MatPlotAgentResponse(*expected, rationale=None)
    got = parse_matplotagent_prompt_response(content)
    if math.isnan(expected[0]):
        assert math.isnan(got[0]) and got[1] == expected[1]
    else:
        assert got == expected
