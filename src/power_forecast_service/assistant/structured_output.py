"""vLLM有限数组约束的等价传输；业务验收始终使用原schema。"""

from copy import deepcopy
from itertools import permutations
from math import perm

from jsonschema import Draft202012Validator

SCHEMA_TRANSPORT_VERSION = "finite-unique-arrays-v1"
# 可选：约束解码按properties顺序生成字段；按字母序排列可与sort_keys序列化的训练目标对齐。
# 只改变对象键的生成顺序，JSON对象语义不变；默认关闭，保持原传输字节不变。
SORTED_PROPERTIES_VERSION = "sorted-properties-v1"
MAX_ENUM_SEQUENCES = 326
MAX_SCHEMA_SEQUENCES = 2048

# 只遍历schema位置。enum/default/examples中的JSON对象是数据，不能当schema改写。
_SCHEMA_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_SCHEMA_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_VALUES = {"items", "additionalItems", "contains", "additionalProperties", "unevaluatedItems",
                  "unevaluatedProperties", "not", "if", "then", "else", "propertyNames", "contentSchema"}
_ANNOTATIONS = {"title", "description", "$comment", "default", "examples", "deprecated", "readOnly", "writeOnly"}


class SchemaCompatibilityError(ValueError):
    """拒绝无法在有限预算内等价转换的schema，不删除约束继续请求。"""


def vllm_schema(schema, *, sort_properties=False):
    """不修改调用者对象；完整后端支持还须在实际引擎解释器中编译检查。

    sort_properties=True 只把schema位置上的properties按键名排序；enum/default等数据不动。"""
    Draft202012Validator.check_schema(schema)
    sequence_count = 0

    def transform(node):
        nonlocal sequence_count
        if isinstance(node, bool):
            return node
        result = deepcopy(node)
        if result.get("uniqueItems") is True:
            allowed = {"type", "items", "minItems", "maxItems", "uniqueItems"} | _ANNOTATIONS
            items = result.get("items", {})
            if (set(result) - allowed or result.get("type") != "array" or not isinstance(items, dict)
                    or set(items) != {"type", "enum"} or items.get("type") != "string"):
                raise SchemaCompatibilityError("schema_unique_array_not_finite")
            names = items["enum"]
            if not all(isinstance(name, str) for name in names) or len(names) != len(set(names)):
                raise SchemaCompatibilityError("schema_unique_array_domain_invalid")
            minimum = result.get("minItems", 0)
            maximum = min(result.get("maxItems", len(names)), len(names))
            # 先计算数量再分配，避免较大候选域造成阶乘枚举。
            count = 0
            for length in range(minimum, maximum + 1):
                count += perm(len(names), length)
                if count > MAX_ENUM_SEQUENCES:
                    raise SchemaCompatibilityError("schema_unique_array_budget")
            sequence_count += count
            if sequence_count > MAX_SCHEMA_SEQUENCES:
                raise SchemaCompatibilityError("schema_total_enum_budget")
            annotations = {key: value for key, value in result.items() if key in _ANNOTATIONS}
            if not count:
                return {**annotations, "not": {}}  # 原约束无解，保持空语言。
            return {**annotations, "enum": [list(items) for length in range(minimum, maximum + 1)
                                            for items in permutations(names, length)]}
        if result.get("uniqueItems") is False:
            result.pop("uniqueItems")
        for key, value in list(result.items()):
            if key in _SCHEMA_MAPS:
                children = sorted(value.items()) if sort_properties and key == "properties" else value.items()
                result[key] = {name: transform(child) for name, child in children}
            elif key in _SCHEMA_LISTS:
                result[key] = [transform(child) for child in value]
            elif key in _SCHEMA_VALUES:
                result[key] = [transform(child) for child in value] if isinstance(value, list) else transform(value)
            elif key == "dependencies":
                result[key] = {name: child if isinstance(child, list) else transform(child)
                               for name, child in value.items()}
        return result

    return transform(schema)
