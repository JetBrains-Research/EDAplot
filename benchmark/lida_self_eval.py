import json
import logging
from dataclasses import dataclass
from typing import Literal

import lida

from edaplot.spec_utils import SpecType

logger = logging.getLogger(__name__)


@dataclass
class LIDASelfEvalScore:
    score: float
    dimension: str
    rationale: str | None


@dataclass
class LIDASelfEvalOutput:
    # N.B. lida doesn't expose the prompts/messages used via the api :(
    prompt: str
    library: str
    code_or_spec: str
    raw_scores: list[dict]

    def parse_scores(self) -> list[LIDASelfEvalScore]:
        scores = []
        for raw_score in self.raw_scores:
            score = LIDASelfEvalScore(
                score=float(raw_score.get("score", "nan")),
                dimension=raw_score.get("dimension", "unknown"),
                rationale=raw_score.get("rationale"),
            )
            scores.append(score)
        return scores


@dataclass
class LIDASelfEvalConfig:
    eval_model_name: str = "gpt-4o-2024-11-20"  # "gpt-4o-mini-2024-07-18"
    eval_temperature: float = 0.0
    lida_library: str = "altair"
    lida_use_cache: bool = False
    lida_eval_choice: Literal["code", "spec"] = "code"


def run_lida_self_eval(
    code_or_spec: str | SpecType, prompt: str, config: LIDASelfEvalConfig, api_key: str | None = None
) -> LIDASelfEvalOutput:
    lida_manager = lida.Manager(lida.llm("openai", api_key=api_key))
    textgen_config = lida.TextGenerationConfig(
        n=1, temperature=config.eval_temperature, model=config.eval_model_name, use_cache=config.lida_use_cache
    )
    goal = lida.components.Goal(question=prompt, visualization=prompt, rationale="")
    if not isinstance(code_or_spec, str):
        # The prompt looks like: "... given the goal and code below in {library}..."
        code = json.dumps(code_or_spec, indent=2)
        library = "Vega-Lite JSON"
    else:
        code = code_or_spec
        library = config.lida_library
    evaluations: list[list[dict]] = lida_manager.evaluate(
        code=code, goal=goal, textgen_config=textgen_config, library=library
    )
    assert len(evaluations) == 1, f"Expected 1 evaluation, got {len(evaluations)}: {evaluations}"
    return LIDASelfEvalOutput(
        prompt=prompt,
        library=library,
        code_or_spec=code,
        raw_scores=evaluations[0],
    )
