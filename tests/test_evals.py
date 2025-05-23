from benchmark.evals.eval_runner import read_eval_inputs
from benchmark.evals.eval_types import (
    ActionOutput,
    ActionType,
    CheckType,
    EvalInput,
    EvalOutput,
    InputAction,
    InputData,
    TargetCheck,
)
from edaplot.vega import MessageType
from edaplot.vega_chat.vega_chat import ChatSession


def test_read_evals() -> None:
    inputs = read_eval_inputs()
    assert len(inputs) > 0


def test_read_write_output() -> None:
    inp = EvalInput(
        id="input",
        data=InputData(path="path.csv"),
        actions=[
            InputAction(
                action_type=ActionType.USER_UTTERANCE,
                action_kwargs={"user_utterance": "text"},
                checks=[TargetCheck(check_type=CheckType.MARK, check_kwargs={"mark_type": ["point"]})],
            )
        ],
    )
    m = ChatSession.create_message("test", MessageType.USER)
    m.spec = {"mark": "bar"}
    expected = EvalOutput(
        input=inp,
        action_outputs=[
            ActionOutput(
                vega_chat_messages=[m],
            )
        ],
        check_results=[{CheckType.MARK: {"check_mark": 1.0}}],
    )
    got = EvalOutput.from_json(expected.to_json())
    assert got == expected
