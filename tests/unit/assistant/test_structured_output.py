"""等价性、预算拒绝与schema/数据位置边界；不声称实际引擎编译。"""
from copy import deepcopy
import json
from itertools import product

from jsonschema import Draft202012Validator
import pytest

from power_forecast_service.assistant.structured_output import SchemaCompatibilityError, vllm_schema


@pytest.mark.parametrize("bounds", [{}, {"minItems": 1, "maxItems": 2}, {"minItems": 4}, {"maxItems": 0}])
def test_finite_language_equivalence_with_negative_values(bounds):
    original = {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": ["a", "b", "c"]}, **bounds}
    before = deepcopy(original)
    wire = vllm_schema(original)
    left, right = Draft202012Validator(original), Draft202012Validator(wire)
    candidates = [None, True, "a", {}, [None], [1]]
    candidates += [list(items) for length in range(5) for items in product(["a", "b", "c", "unknown"], repeat=length)]
    assert all(left.is_valid(value) == right.is_valid(value) for value in candidates)
    assert original == before


@pytest.mark.parametrize("schema", [
    {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
    {"uniqueItems": True, "items": {"type": "string", "enum": ["a"]}},
    {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": list("abcdef")}},
    {"type": "array", "uniqueItems": True, "items": {"type": "number", "enum": [1, 2]}},
    {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": ["a"]}, "contains": {"const": "a"}},
])
def test_unsupported_unique_domain_is_rejected_without_weakening(schema):
    with pytest.raises(SchemaCompatibilityError):
        vllm_schema(schema)


def test_data_inside_enum_and_default_are_not_schemas():
    literal = {"uniqueItems": True, "type": "array", "items": {"type": "string"}}
    schema = {"enum": [literal], "default": literal, "examples": [literal]}
    assert vllm_schema(schema) == schema


def test_refs_conditionals_and_false_unique_keep_their_language():
    schema = {"$defs": {"ids": {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": ["a", "b"]}}},
              "if": {"type": "array"}, "then": {"$ref": "#/$defs/ids"}, "else": {"type": "boolean"},
              "uniqueItems": False}
    wire = vllm_schema(schema)
    for value in ([], ["a"], ["a", "a"], ["b", "a"], ["unknown"], True, {}, "a"):
        assert Draft202012Validator(schema).is_valid(value) == Draft202012Validator(wire).is_valid(value)


def test_large_domain_with_small_length_is_bounded_and_exact():
    schema = {"type": "array", "uniqueItems": True, "maxItems": 1,
              "items": {"type": "string", "enum": [str(n) for n in range(100)]}}
    assert len(vllm_schema(schema)["enum"]) == 101


def test_aggregate_enumeration_is_bounded():
    child = {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": list("abcde")}}
    with pytest.raises(SchemaCompatibilityError, match="total_enum_budget"):
        vllm_schema({"type": "object", "properties": {str(n): child for n in range(7)}})


def test_sorted_properties_is_default_off_and_only_reorders_schema_positions():
    data = {"properties": {"z": 1, "a": 2}}
    schema = {"type": "object", "properties": {
        "requests": {"type": "array", "items": {"type": "object", "properties": {
            "resolution": {"type": "string"}, "presentation": {"enum": [data]}, "request": {"type": "string"}}}},
        "assessment": {"type": "string", "default": data}}, "required": ["requests", "assessment"]}
    before = deepcopy(schema)
    # 默认关闭：传输schema与原有行为逐字一致，包括键顺序。
    assert list(vllm_schema(schema)["properties"]) == ["requests", "assessment"]
    assert json.dumps(vllm_schema(schema)) == json.dumps(schema)
    wire = vllm_schema(schema, sort_properties=True)
    assert list(wire["properties"]) == ["assessment", "requests"]
    assert list(wire["properties"]["requests"]["items"]["properties"]) == ["presentation", "request", "resolution"]
    # enum/default中的JSON对象是数据，键序不能被改写；required列表也不变。
    assert list(wire["properties"]["requests"]["items"]["properties"]["presentation"]["enum"][0]["properties"]) == ["z", "a"]
    assert list(wire["properties"]["assessment"]["default"]["properties"]) == ["z", "a"]
    assert wire["required"] == ["requests", "assessment"]
    assert schema == before and json.dumps(schema) == json.dumps(before)
    left, right = Draft202012Validator(schema), Draft202012Validator(wire)
    for value in ({"requests": [{"resolution": "x", "request": "y", "presentation": data}], "assessment": "a"},
                  {"requests": [{"resolution": 1}], "assessment": "a"}, {"requests": []}, {}):
        assert left.is_valid(value) == right.is_valid(value)


def test_sorted_properties_composes_with_unique_array_expansion():
    schema = {"type": "object", "properties": {"b": {"type": "array", "uniqueItems": True,
        "items": {"type": "string", "enum": ["x", "y"]}}, "a": {"type": "string"}}}
    wire = vllm_schema(schema, sort_properties=True)
    assert list(wire["properties"]) == ["a", "b"]
    assert wire["properties"]["b"] == vllm_schema(schema)["properties"]["b"]
