"""排列可改变前缀，但不得删减证据、改值或改变默认请求字节。"""

from copy import deepcopy
import json
from uuid import UUID

import pytest

from power_forecast_service.assistant.contracts import ContextRef, DraftAnswer, Question
from power_forecast_service.assistant.prompting import answer_messages
from power_forecast_service.assistant.workflow import Assistant, INSTRUCTIONS


def fixture_payload():
    question = Question(question="采用门为什么没有通过？", contexts=[
        ContextRef(kind="engie_import", id=UUID(int=1))])
    evidence = {"stage_requirements": [{"role": "adoption_decision"}],
                "facts": [{"id": "c0.mae", "value": 2.985, "unit": "%"}],
                "records": [{"note": "保留负号和原始含义", "value": -1}],
                "stage_options": [{"role": "adoption_decision"}]}
    documents = [{"id": "dev", "text": "完整中文原文", "score": 0.4},
                 {"id": "final", "text": "正式结果不能反推开发门"}]
    return question, evidence, documents


def test_default_messages_preserve_original_bytes():
    question, evidence, documents = fixture_payload()
    schema = DraftAnswer.model_json_schema()
    expected = [("system", INSTRUCTIONS + "\nJSON schema:" + json.dumps(schema, ensure_ascii=False)),
                ("user", json.dumps({"question": question.question,
                    "contexts": [c.model_dump(mode="json") for c in question.contexts],
                    "evidence": evidence, "documents": [{k: v for k, v in d.items() if k != "score"}
                                                           for d in documents]}, ensure_ascii=False, default=str))]
    assert answer_messages(question, evidence, documents, INSTRUCTIONS, schema) == expected


def test_evidence_first_keeps_every_value_and_does_not_mutate_input():
    question, evidence, documents = fixture_payload()
    before = deepcopy((evidence, documents))
    args = (question, evidence, documents, INSTRUCTIONS, DraftAnswer.model_json_schema())
    original = answer_messages(*args)
    ordered = answer_messages(*args, layout="evidence_first")
    assert original[0] == ordered[0]
    assert json.loads(original[1][1]) == json.loads(ordered[1][1])
    assert list(json.loads(ordered[1][1])) == ["contexts", "evidence", "documents", "question"]
    assert list(json.loads(ordered[1][1])["evidence"])[-1] == "stage_requirements"
    assert (evidence, documents) == before


def test_changed_evidence_changes_prefix_before_question():
    question, evidence, documents = fixture_payload()
    schema = DraftAnswer.model_json_schema()
    first = answer_messages(question, evidence, documents, INSTRUCTIONS, schema, layout="evidence_first")
    evidence["facts"][0]["value"] = 3.001
    changed = answer_messages(question, evidence, documents, INSTRUCTIONS, schema, layout="evidence_first")
    assert first[1][1] != changed[1][1]
    assert changed[1][1].index("3.001") < changed[1][1].index('"question"')


def test_invalid_layout_is_rejected_before_creating_executor():
    with pytest.raises(ValueError, match="layout"):
        Assistant(None, prompt_layout="unknown")
