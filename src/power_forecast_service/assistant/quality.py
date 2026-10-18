"""已观察到的正文反例检查；不声称验证任意自然语言的蕴含关系。"""

import re

from .contracts import AssistantError

_CHINESE = "零〇一二两三四五六七八九十百千万亿"


def chinese_integer(value):
    """仅解释十进制中文整数；调用方限定为带明确业务标签的数量。"""
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    small = {"十": 10, "百": 100, "千": 1000}
    total = section = number = 0
    if all(char in digits for char in value):
        return int("".join(str(digits[char]) for char in value))
    for char in value:
        if char in digits:
            number = digits[char]
        elif char in small:
            section += (number or 1) * small[char]
            number = 0
        elif char == "万":
            total += (section + number) * 10000
            section = number = 0
        elif char == "亿":
            total = (total + section + number) * 100000000
            section = number = 0
        else:
            raise ValueError("Not a supported Chinese integer")
    return total + section + number


def text_findings(text, facts):
    """返回可定位的反例，不改写文本；引用的错误说法不当作模型自己的断言。"""
    findings = []
    prose = re.sub(r"[“\"][^”\"]*[”\"]|‘[^’]*’", "", text)
    if re.search(r"(?:为|是|[：:])\s*(?:\*\*\s*)?(?:个百分点|分钟|小时|kW|%|次)(?=[，。；！？\s]|$)", prose) or re.search(r"(?<!\*)\*{4}(?!\*)", prose):
        findings.append(AssistantError("answer_prose_incomplete", "正文存在空白数值或空强调，请完整解释，不用事实附件替代正文。"))
    counts = {f["value"] for f in facts if f["id"].endswith(".planned") and type(f["value"]) in {int, float}}
    if len(counts) == 1:
        for match in re.finditer(r"计划起报(?:总数|次数|数量|数)?(?:为|是|共|有|计|[：:])?\s*([" + _CHINESE + r"]+)次", prose):
            if chinese_integer(match[1]) != next(iter(counts)):
                findings.append(AssistantError("answer_chinese_count_conflict", "计划起报的中文数量与所选对象事实矛盾；请引用该对象的真实数量。"))
    adopted = [f["value"] for f in facts if f["id"].endswith(".adopted")]
    if adopted and all(value is False for value in adopted):
        for sentence in re.split(r"[。！？;；\n]", prose):
            if re.search(r"默认(?:模型)?(?:已经|已)(?:被)?(?:替换|切换|改为)", sentence) and not re.search(r"(?:并非|不是|不能说|不能断言|不可断言|不代表|不意味着|错误|不成立|不对|未曾|没有)", sentence):
                findings.append(AssistantError("answer_adoption_contradiction", "所选对象的采用状态为否，不能断言默认模型已经替换。"))
    return findings


def validate_prose(draft, evidence):
    """阶段回答仅使用对应对象的事实，避免拿另一对象的状态判断正文。"""
    parts = [(claim.text, str(claim.object_id)) for claim in draft.stage_claims] or [(draft.answer, None)]
    for text, identity in parts:
        facts = [fact for fact in evidence["facts"] if identity is None or fact.get("object_id") == identity]
        errors = text_findings(text, facts)
        if errors:
            raise errors[0]
