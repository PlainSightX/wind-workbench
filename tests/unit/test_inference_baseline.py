"""只验证回放工具的边界；不代替真实模型或业务语义评估。"""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('inference_baseline', Path(__file__).parents[2]/'tools/diagnostics/inference_baseline.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_workload_identity_and_refuse_overwrite(tmp_path):
    data = {'cases':[{'messages':[{'role':'user','content':'完整中文上下文'}]}]}
    data['sha256'] = module.digest(data)
    path = tmp_path/'workload.json'
    module.save(path,data)
    assert module.load_workload(path)==data
    with pytest.raises(FileExistsError):
        module.save(path,data)
    data['cases'][0]['messages'][0]['content']='被篡改'
    path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='identity'):
        module.load_workload(path)


@pytest.mark.parametrize('text', ['sk-'+'a'*30, r'D:\private\file', 'Bearer '+'b'*30])
def test_payload_does_not_export_obvious_private_values(text):
    with pytest.raises(ValueError,match='private-data'):
        module.public_payload([{'role':'user','content':text}])


def test_api_key_name_without_value_is_not_a_secret():
    module.public_payload([{'role':'user','content':'不要泄露DEEPSEEK_API_KEY'}])
    module.public_payload([{'role':'user','content':'公开文档入口 http://127.0.0.1:18000/#predict'}])


@pytest.mark.parametrize('left,right,expected', [([1,2],[1,2,3],2),([1,2],[1,3],1),([],[],0),([1],[2],0)])
def test_prefix_is_exact_token_identity(left,right,expected):
    assert module.common_prefix(left,right)==expected


def test_sse_ignores_comments_and_keeps_chinese():
    assert module.decode_sse(': ping') is None
    assert module.decode_sse('data: [DONE]')=='done'
    assert module.decode_sse('data: {"content":"中文协作正常"}')=={'content':'中文协作正常'}


def test_chat_template_requests_flat_complete_token_ids():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]['content']=='完整上下文'
            assert kwargs=={'tokenize':True,'add_generation_prompt':True,'return_dict':False}
            return [1,2,3]
    assert module.chat_token_ids(Tokenizer(),[{'role':'user','content':'完整上下文'}])==[1,2,3]


def test_truncated_generation_cannot_pass_quality():
    assert module.check_answer('{}',{},'length')['machine_pass'] is False
