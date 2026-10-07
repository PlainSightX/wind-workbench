"""完整证据的请求排列；原排列保持字节兼容，可选排列用于前缀复用对照。"""

import json

PROMPT_LAYOUTS = ("original", "evidence_first")


def answer_messages(question, evidence, documents, instructions, schema, *, layout="original", schema_placement="system"):
    """仅调整对象键的顺序，保留值、数组顺序、引用及全部文档。"""
    if layout not in PROMPT_LAYOUTS:
        raise ValueError("Unknown assistant prompt layout")
    if schema_placement not in ("system", "user_tail"):
        raise ValueError("Unknown schema placement")
    contexts = [context.model_dump(mode="json") for context in question.contexts]
    docs = [{key: value for key, value in document.items() if key != "score"}
            for document in documents]
    if layout == "original":
        prompt = {"question": question.question, "contexts": contexts,
                  "evidence": evidence, "documents": docs}
    else:
        # 阶段要求随问题变化，放到证据对象末尾；每次仍消费本次完整的阶段要求。
        ordered = {key: value for key, value in evidence.items() if key != "stage_requirements"}
        if "stage_requirements" in evidence:
            ordered["stage_requirements"] = evidence["stage_requirements"]
        prompt = {"contexts": contexts, "evidence": ordered,
                  "documents": docs, "question": question.question}
    # 请求专属schema放在完整证据之后；不能让逐题变化的语法抢占公共前缀。
    system = instructions
    if schema_placement == "user_tail":
        prompt["output_schema"] = schema
    else:
        system += "\nJSON schema:" + json.dumps(schema, ensure_ascii=False)
    return [
        ("system", system),
        ("user", json.dumps(prompt, ensure_ascii=False, default=str)),
    ]
