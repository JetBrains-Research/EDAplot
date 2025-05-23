import enum
import logging
import re
from typing import Any

import commentjson

logger = logging.getLogger(__name__)


class PromptTag(enum.StrEnum):
    @property
    def open_tag(self) -> str:
        return f"<{self.value}>"

    @property
    def close_tag(self) -> str:
        return f"</{self.value}>"

    def wrap(self, content: str) -> str:
        return f"{self.open_tag}{content}{self.close_tag}"


def extract_tag_content(s: str, tag: str) -> list[str]:
    m = re.findall(rf"<{tag}>(.*?)<(?:/{tag}|{tag})>", s, re.DOTALL)
    if len(m) > 0:
        return m
    return []


def extract_json_tags(content: str, tag: PromptTag) -> list[Any]:
    # Sometimes multiple comma-separated JSONs are generated which is not valid JSON.
    # Sometimes multiple <tag> separated JSONs are generated.
    #  Hacky solution: make it into an array to get a valid JSON for parsing
    # LLMs sometimes generate JSON with comments, so use the commentjson package instead
    json_parts = extract_tag_content(content, tag)
    if len(json_parts) == 0:
        return []
    json_part = ",".join(json_parts)
    try:
        json_dicts = commentjson.loads(f"[{json_part}]")
    except ValueError as e:
        logger.error("extracting model response jsons", exc_info=e)
        raise ValueError(f"Invalid json formatting!")
    if len(json_dicts) == 1:
        if isinstance(json_dicts[0], list):
            json_dicts = json_dicts[0]
    return json_dicts


def print_prompt_helper(title: str, prompt: str) -> None:
    bar = "-" * 16
    # > A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text.
    n_tokens_approx = len(prompt) / 4
    print(f"{title} (length={len(prompt)}, approx tokens={int(n_tokens_approx)})\n{bar}\n{prompt}\n{bar}\n")
