"""明确反例与相邻合法表达；这些用例不证明完整语义正确性。"""

import pytest

from power_forecast_service.assistant.quality import chinese_integer, text_findings


FACTS = [{"id": "other.planned", "value": 52560}, {"id": "other.adopted", "value": False}]


@pytest.mark.parametrize("text,code", [
    ("改善为  个百分点。", "answer_prose_incomplete"),
    ("延迟为 分钟。", "answer_prose_incomplete"),
    ("改善是****。", "answer_prose_incomplete"),
    ("计划起报为五千二百五十六次。", "answer_chinese_count_conflict"),
    ("开发采用门未通过，默认模型已替换。", "answer_adoption_contradiction"),
])
def test_observed_bad_prose_is_not_fixed_by_correct_fact_list(text, code):
    assert code in {error.code for error in text_findings(text, FACTS)}


@pytest.mark.parametrize("text", [
    "计划起报为五万二千五百六十次。",
    "起报为{{other.planned}}，这里没有数值空位。",
    "延迟为{{other.lag}}分钟，属于模拟假设。",
    "默认模型没有替换。",
    "不能说默认模型已替换。",
    "不能断言默认模型已经替换。",
    "“默认模型已替换”的说法不成立。",
    "默认模型并非已经替换。",
    "方法使用三次比较，二十分钟延迟是模拟假设。",
])
def test_nearby_legal_explanations_are_not_blanket_rejected(text):
    assert not text_findings(text, FACTS)


@pytest.mark.parametrize("text,value", [("十二", 12), ("二十", 20), ("一百零三", 103),
                                       ("五万二千五百六十", 52560), ("二〇一五", 2015), ("一亿零三万", 100030000)])
def test_supported_decimal_chinese_integers(text, value):
    assert chinese_integer(text) == value


def test_count_or_state_cannot_be_guessed_for_ambiguous_objects():
    assert not text_findings("计划起报为十次。", [{"id": "a.planned", "value": 10}, {"id": "b.planned", "value": 20}])
    assert not text_findings("默认模型已替换。", [{"id": "a.adopted", "value": True}])
