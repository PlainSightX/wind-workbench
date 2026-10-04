"""事实、引用、阶段约束与正文渲染；无网络、数据库或模型调用。"""

import re

from .contracts import AssistantError
from .stages import validate_stage_claims, stage_body


FACT_TOKEN = re.compile(r"\{\{([^{}]+)\}\}")
FACT_SIGN = re.compile(r"[+\-−﹣－＋±∓]\s*(?:[*`_~(（\[]\s*)*$")


def fact_unit_suffix(text, units):
    """占位符是完整量纲；后接的Latin标记不能绕过已知单位列表。"""
    # 中文连词、句读和Markdown结束符仍是正文；紧接Latin词或量纲运算符具有单位歧义，
    # 应由一次repair删除多余后缀，不能把未知GW/kg悄悄当作可信事实的一部分。
    wrappers = r"[\s*`_~()（）\[\]]*"
    unknown = r"[A-Za-zµμΩ°℃℉][A-Za-z0-9µμΩ°℃℉/%·*^²³⁻+\-]*"
    operation = r"[/·^²³⁻][^\s，。；,;！？!?()\[\]{}]*"
    return re.match(wrappers + "(" + unit_pattern(units) + "|" + unknown + "|" + operation + ")", text)


def validate_fact_boundaries(body, facts, units):
    """先检查完整事实周边，再去占位符；否则外加负号会从数字检查中消失。"""
    for token in FACT_TOKEN.finditer(body):
        fact = facts[token[1]]
        if type(fact["value"]) in {int, float}:
            prefix = body[:token.start()]
            sign = FACT_SIGN.search(prefix)
            # 行首Markdown列表的'- ' / '+ '不是一元运算；句内和紧贴负号必须拒绝。
            bullet = re.search(r"(?:^|\n)[ \t]*[+\-][ \t]+(?:[*`_~(（\[][ \t]*)*$", prefix)
            if sign and not bullet:
                raise AssistantError("answer_fact_sign_conflict", "占位符已含原事实符号，不可另加符号：" + token[1])
        explicit = fact_unit_suffix(body[token.end():], units)
        if explicit and (explicit[1] != fact["unit"] or
                         fact_unit_suffix(body[token.end() + explicit.end():], units)):
            raise AssistantError("answer_fact_unit_conflict", "占位符后单位与原事实不一致：" + token[1])


def unit_pattern(units):
    known = "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True) if unit)
    # kW/h不能只消费kW前缀；只识别显式复合后缀，不尝试理解自然语言单位换算。
    return r"(?:" + known + r")(?:[ \t]*[/·*^][ \t]*[^\s，。；,;！？!?()\[\]{}]*)*(?![A-Za-z])"


def numeric_tokens(text, units):
    """保留符号和显式单位；不能把12.5 kW与-12.5%视为同一证据。"""
    text = text.translate(str.maketrans({"−": "-", "﹣": "-", "－": "-"}))
    pattern = r"(?<![\d.])([+\-]?\d+(?:\.\d+)?)(?:\s*(" + unit_pattern(units) + r"))?"
    return {(number, re.sub(r"[ \t]+", "", unit)) for number, unit in re.findall(pattern, text)}


def validate_answer(draft, evidence, documents, question=""):
    facts = {f["id"]: f for f in evidence["facts"]}
    docs = {d["id"]: d for d in documents}
    # 规范化已知的无歧义占位符写法；不能把未知ID猜成最近的事实。
    normalize_fact_tokens = lambda text: re.sub(r"\{\{\s*(?:fact_id:\s*)?([^{}]+?)\s*\}\}", lambda m:"{{"+m.group(1).strip()+"}}", text)
    draft.answer = normalize_fact_tokens(draft.answer)
    for claim in draft.stage_claims:
        claim.text = normalize_fact_tokens(claim.text)
    # 阶段片段是正文的唯一输入，避免要求模型维护两份逐字一致的自由文本。
    body = "\n".join(c.text for c in draft.stage_claims) if draft.stage_claims else draft.answer
    if not body.strip():
        raise AssistantError("answer_body_missing")
    referenced = re.findall(r"\{\{([^{}]+)\}\}", body)
    if any(f not in facts for f in draft.fact_ids + referenced) or set(referenced) - set(draft.fact_ids):
        raise AssistantError("answer_fact_invalid")
    units = {f.get("unit", "") for f in facts.values()} | {"%", "kW", "MW", "W", "kWh", "MWh", "分钟", "小时", "秒", "次", "minutes", "min"}
    validate_fact_boundaries(body, facts, units)
    # 页面是纯文本，不会把Markdown的'- '解释为列表；规范成可见圆点避免误读成负值。
    normalize_bullets = lambda text: re.sub(r"(?m)^([ \t]*)[+\-]([ \t]+)(?=(?:[*`_~(（\[][ \t]*)*\{\{)", r"\1•\2", text)
    draft.answer = normalize_bullets(draft.answer)
    for claim in draft.stage_claims:
        claim.text = normalize_bullets(claim.text)
    # 未在正文用到的facts不展示、不计入评分，防止附上正确列表却没有回答问题。
    draft.fact_ids = list(dict.fromkeys(referenced))
    if any(c not in docs for c in draft.citations) or set(draft.quotes) - set(draft.citations):
        raise AssistantError("answer_citation_invalid")
    normalize = lambda value: re.sub(r"[\s*`]+", "", value)
    for key, quote in draft.quotes.items():
        if len(quote.strip()) < 4 or normalize(quote) not in normalize(docs[key]["text"]):
            raise AssistantError("answer_quote_invalid", "引文不属于此段或并非连续原文：" + key)
    if draft.citations and any(c not in draft.quotes for c in draft.citations):
        raise AssistantError("answer_quote_missing")
    raw = re.sub(r"\{\{[^{}]+\}\}", "", body)
    # 技术名字允许数字；其他自由书写数字不作为可信指标输出。
    raw = re.sub(r"(?<![a-zA-Z0-9_])(?:Q[1-4]|L[12])(?![a-zA-Z0-9_])", "", raw)
    model_names = {name for record in evidence["records"] for name in record["models"]}
    for name in sorted(model_names, key=len, reverse=True):
        raw = raw.replace(name, "")
    # 原文方法数字可被复述；指标占位符仍绑定完整身份。此规则不证明句子蕴含关系。
    numbers = lambda text: numeric_tokens(text, units)
    # citation定位的是完整原文段落，quotes只负责确认引用确实落在该段。
    # 同段的参数数字不要求模型重复抄一遍整段；页面始终展示真实完整段落。
    supported = set(numbers(" ".join(docs[c]["text"] for c in draft.citations)))
    # 时间范围是对象元数据，不是指标；情景时刻可原样复述用户问题。
    metadata = " ".join(str(r.get(k,"")) for r in evidence["records"] for k in ("quarter","training_label_available","purpose","horizon_minutes"))
    supported.update(numbers(metadata))
    # horizon_minutes字段自带分钟语义；不把元数据60随意变成60 kW或60%。
    for record in evidence["records"]:
        if record.get("horizon_minutes") is not None:
            supported.update(numbers(str(record["horizon_minutes"]) + "分钟"))
            supported.update(numbers(str(record["horizon_minutes"]) + " minutes"))
    for clock in re.findall(r"(?<!\d)\d{1,2}:\d{2}(?!\d)", question):
        raw = raw.replace(clock, "")
    numeric_facts = [facts[f] for f in referenced if type(facts[f]["value"]) in {int,float}]
    def matched(token):
        value, unit = token
        digits = len(value.split(".")[1]) if "." in value else 0
        return any(unit == fact["unit"] and abs(float(value)-round(fact["value"],digits)) < 1e-8 for fact in numeric_facts)
    unsupported = sorted(value + unit for value, unit in numbers(raw) - supported if not matched((value, unit)))
    if unsupported:
        raise AssistantError("answer_number_unbound", "缺少对应引用或facts绑定的数字：" + ", ".join(unsupported))
    if draft.status == "answered" and not (draft.fact_ids or draft.citations):
        raise AssistantError("answer_evidence_missing")
    validate_stage_claims(draft, evidence)
    def display(match):
        fact = facts[match.group(1)]
        value = fact["value"]
        if isinstance(value, bool):
            value = "是" if value else "否"
        elif isinstance(value, (int, float)):
            value = f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{fact['label']}：{value} {fact['unit']}".strip()
    rendered = re.sub(r"\{\{([^{}]+)\}\}", display, stage_body(draft, evidence))
    rendered = re.sub(r"(kW|%|次|UTC)\s*\1", r"\1", rendered)
    return {"status": draft.status, "answer": rendered,
        "facts": [facts[f] for f in dict.fromkeys(draft.fact_ids)],
        "citations": [{"id": c, "title": docs[c]["title"], "revision": docs[c]["source_sha256"],
            "quote": docs[c]["text"], "url": f"/assistant/documents/{c}"} for c in dict.fromkeys(draft.citations)]}
