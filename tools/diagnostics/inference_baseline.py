"""完整请求回放基线：记录引擎耗时与原业务验收，不改默认provider。"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'Qwen/Qwen3-4B-Instruct-2507-FP8'
REVISION = '8591804019c8b22094c3b5b4454e0edc05dffc98'
SELECTED = ('N2', 'N4', 'N7', 'M1', 'M4', 'B4')


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')


def load_workload(path):
    value = json.loads(path.read_bytes())
    expected = value.pop('sha256')
    if digest(value) != expected:
        raise ValueError('Workload identity mismatch')
    value['sha256'] = expected
    return value


def public_payload(messages):
    # 这是本轮选定材料的额外检查，不声称通用secret扫描。异常不回显正文。
    text = json.dumps(messages, ensure_ascii=False)
    if re.search(r'sk-[A-Za-z0-9_-]{16,}|(?<![A-Za-z])[A-Za-z]:[\\/]|Bearer\s+[A-Za-z0-9_-]{16,}', text):
        raise ValueError('Selected payload requires private-data review')


async def capture(output):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from power_forecast_service.assistant.contracts import ContextRef, Question
    from power_forecast_service.assistant.workflow import Assistant, PROMPT_VERSION
    from power_forecast_service.assistant.retrieval import corpus
    from power_forecast_service.settings import Settings
    from power_forecast_service.storage.database import make_async_engine

    contract_path = ROOT / 'docs/results/wind-assistant-c-20260923/questions.json'
    contract = json.loads(contract_path.read_bytes())
    cases = {c['id']: c for c in contract['cases']}
    engine = make_async_engine(Settings.from_environment())
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    class Captured(Exception):
        pass

    class CaptureAssistant(Assistant):
        async def call(self, state, schema, messages):
            self.captured = [{'role': role, 'content': content} for role, content in messages]
            raise Captured()

    assistant = CaptureAssistant(sessions)
    rows = []
    try:
        for key in SELECTED:
            item = cases[key]
            question = Question(question=item['question'], contexts=[ContextRef.model_validate(c) for c in item['contexts']])
            state = {'question': question, 'trace': {'tools': [], 'documents': []}, 'strategy': 'keyword'}
            state.update(await assistant.evidence(state))
            try:
                await assistant.answer(state)
            except Captured:
                pass
            else:
                raise ValueError('Capture unexpectedly executed model')
            public_payload(assistant.captured)
            rows.append({'id':key, 'messages':assistant.captured,
                         'evidence':state['evidence'], 'documents':state['documents'],
                         'question':item['question'], 'required_facts':item['required_facts'],
                         'semantic_rubric':item['semantic_rubric']})
    finally:
        assistant.close()
        await engine.dispose()
    value = {'schema_version':1, 'created_at':datetime.now(UTC).isoformat(),
             'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'workflow_sha256':sha256((ROOT/'src/power_forecast_service/assistant/workflow.py').read_bytes()).hexdigest(),
             'prompt_version':PROMPT_VERSION, 'corpus_sha256':corpus()['sha256'],
             'selection':'Before inference: numeric comparison, coverage, interval limits, adoption boundary, time boundary and false-premise refusal; known development questions, not unseen evaluation',
             'assembly':'Current Assistant.evidence/answer, read-only real PG, keyword retrieval, complete first-call messages; no audit write or model call during capture',
             'model_candidate':MODEL, 'model_revision':REVISION,
             'max_tokens':1700, 'temperature':0, 'seed':42,
             'quality':'Original schema/semantic binding checks plus required facts; independent natural-language rubric review remains required',
             'cases':rows}
    value['sha256'] = digest(value)
    save(output, value)
    print(json.dumps({'cases':len(rows),'sha256':value['sha256'],'model_calls':0,'complete_messages':True}))


def download(directory, proxy, weights):
    """下载固定revision并校验LFS权重，不执行远程代码，不读取HF凭据。"""
    opener = build_opener(ProxyHandler({'https':proxy, 'http':proxy}) if proxy else ProxyHandler({}))
    base = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/'
    with opener.open(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true', timeout=45) as reply:
        metadata = json.load(reply)
    if metadata['sha'] != REVISION:
        raise ValueError('Model revision mismatch')
    directory.mkdir(parents=True, exist_ok=True)
    selected = {'config.json','tokenizer.json','tokenizer_config.json','generation_config.json','merges.txt','vocab.json','LICENSE'}
    if weights:
        selected.add('model.safetensors')
    manifest = {'model':MODEL,'revision':REVISION,'files':{}}
    for item in metadata['siblings']:
        name = item['rfilename']
        if name not in selected:
            continue
        target = directory / name
        expected = item.get('lfs',{}).get('sha256')
        if target.exists():
            actual = file_digest(target)
            if target.stat().st_size != item['size'] or (expected and expected != actual):
                raise ValueError('Existing model file identity mismatch: '+name)
        else:
            partial = target.with_suffix(target.suffix+'.partial')
            offset = partial.stat().st_size if partial.exists() else 0
            from urllib.request import Request
            request = Request(base+name, headers={'Range':f'bytes={offset}-'} if offset else {})
            with opener.open(request, timeout=120) as reply:
                resume = offset and reply.status == 206
                with partial.open('ab' if resume else 'wb') as stream:
                    count = offset if resume else 0
                    last = time.monotonic()
                    while block := reply.read(4*1024*1024):
                        stream.write(block)
                        count += len(block)
                        if time.monotonic()-last > 25:
                            print(json.dumps({'file':name,'downloaded_bytes':count,'total_bytes':item['size']}),flush=True)
                            last = time.monotonic()
            actual = file_digest(partial)
            if partial.stat().st_size != item['size'] or (expected and expected != actual):
                raise ValueError('Downloaded model identity mismatch: '+name)
            partial.replace(target)
        manifest['files'][name] = {'sha256':actual,'size':item['size'],'upstream_lfs_sha256':expected}
        print(json.dumps({'file':name,'size':item['size'],'verified':True}),flush=True)
    # 同一下载流程的派生清单允许刷新；不覆盖冻结workload或历史实验。
    (directory/'download-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def file_digest(path):
    h = sha256()
    with path.open('rb') as stream:
        while block := stream.read(4*1024*1024):
            h.update(block)
    return h.hexdigest()


def download_native(directory, proxy):
    """大权重交给官方Xet并发传输，缓存仅归本轮，不读取账号token。"""
    import os
    os.environ['HF_HOME'] = str(directory.parent/'hf-cache')
    os.environ['HF_XET_CACHE'] = str(directory.parent/'hf-cache/xet')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    if proxy:
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy
    import httpx
    from huggingface_hub import HfApi, hf_hub_download, set_client_factory
    set_client_factory(lambda: httpx.Client(proxy=proxy, timeout=60, follow_redirects=True, trust_env=False))
    info = HfApi(token=False).model_info(MODEL, revision=REVISION, files_metadata=True)
    target = next(x for x in info.siblings if x.rfilename=='model.safetensors')
    expected = target.lfs.sha256
    print(json.dumps({'method':'official_hf_xet','revision':info.sha,'bytes':target.size}),flush=True)
    path = Path(hf_hub_download(MODEL,'model.safetensors',revision=REVISION,local_dir=directory,token=False))
    actual = file_digest(path)
    if actual != expected or path.stat().st_size != target.size:
        raise ValueError('Official download weight identity mismatch')
    save(directory.parent/'weights-identity.json', {'model':MODEL,'revision':REVISION,'file':'model.safetensors','size':target.size,'sha256':actual,'hub_version':__import__('huggingface_hub').__version__})
    print(json.dumps({'weight_verified':True,'bytes':target.size,'sha256':actual}),flush=True)


def common_prefix(left, right):
    count = 0
    for a,b in zip(left,right):
        if a != b:
            break
        count += 1
    return count


def chat_token_ids(tokenizer, messages, **template_options):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=False, **template_options)


def tokenize(workload_path, model_directory, output):
    from transformers import AutoTokenizer
    value = load_workload(workload_path)
    tokenizer = AutoTokenizer.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False)
    ids = [chat_token_ids(tokenizer, c['messages']) for c in value['cases']]
    config = json.loads((model_directory/'config.json').read_bytes())
    kv_bytes = 2*config['num_hidden_layers']*config['num_key_value_heads']*config['head_dim']*2
    result = {'workload_sha256':value['sha256'],'model':MODEL,'revision':REVISION,
              'tokenizer_files_sha256':{n:file_digest(model_directory/n) for n in ('tokenizer.json','tokenizer_config.json')},
              'kv_bytes_per_token_bf16':kv_bytes,
              'requests':[{'id':c['id'],'input_tokens':len(tokens),'reserved_total_tokens':len(tokens)+value['max_tokens'],'token_ids_sha256':digest(tokens)} for c,tokens in zip(value['cases'],ids)],
              'common_prefix_tokens':[[common_prefix(a,b) for b in ids] for a in ids],
              'truncated':False}
    save(output,result)
    print(json.dumps(result))


def inspect_agent(archive, model_directory, output):
    """只读历史请求字节做选载；不执行第三方源码，不宣称当前模型任务质量。"""
    import zipfile
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False)
    requests = []
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            if not entry.filename.endswith('request.json'):
                continue
            raw = package.read(entry)
            value = json.loads(raw)
            ids = chat_token_ids(tokenizer,value['messages'])
            requests.append({'member':entry.filename,'request_sha256':sha256(raw).hexdigest(),
                             'input_tokens':len(ids),'output_budget':value['max_tokens'],
                             'stream':value['stream'],'original_model':value['model']})
    result = {'archive_sha256':file_digest(archive),'model_tokenizer':MODEL,'revision':REVISION,
              'historical_not_current':True,'truncated':False,'requests':requests,
              'decision':'Defer repair-model consumer: different task/quality contract, long generation and reasoning requirements. Use current Wind bounded evidence-answer requests as first inference consumer; no Agent default change.'}
    save(output,result)
    print(json.dumps({'requests':len(requests),'input_min':min(r['input_tokens'] for r in requests),'input_max':max(r['input_tokens'] for r in requests),'output_max':max(r['output_budget'] for r in requests),'truncated':False,'model_calls':0}))


def decode_sse(line):
    if not line.startswith('data:'):
        return None
    body = line[5:].strip()
    if body == '[DONE]':
        return 'done'
    return json.loads(body)


def check_answer(content, item, finish_reason):
    from power_forecast_service.assistant.contracts import DraftAnswer, AssistantError
    from power_forecast_service.assistant.validation import validate_answer
    from pydantic import ValidationError
    if finish_reason != 'stop':
        return {'machine_pass':False,'error':'incomplete_generation','finish_reason':finish_reason}
    try:
        draft = DraftAnswer.model_validate_json(content)
        answer = validate_answer(draft,item['evidence'],item['documents'],item['question'])
        missing = sorted(set(item['required_facts']) - {f['id'] for f in answer['facts']})
        return {'machine_pass':not missing and answer['status']=='answered','missing_facts':missing,'result':answer,'semantic_review':'pending'}
    except (ValidationError,AssistantError) as exc:
        return {'machine_pass':False,'error':getattr(exc,'code','answer_schema_invalid'),'semantic_review':'not_accepted'}


async def one(client, base_url, value, item, label, repetition):
    started = time.perf_counter()
    first, event_times, content, usage, finish = None, [], '', None, None
    body = {'model':'inference-i1','messages':item['messages'],'max_tokens':value['max_tokens'],
            'temperature':value['temperature'],'seed':value['seed'],
            'response_format':{'type':'json_object'},'stream':True,'stream_options':{'include_usage':True}}
    row = {'case':item['id'],'label':label,'repetition':repetition,'workload_sha256':value['sha256']}
    try:
        async with client.stream('POST',base_url+'/v1/chat/completions',json=body) as reply:
            reply.raise_for_status()
            async for line in reply.aiter_lines():
                chunk = decode_sse(line)
                if chunk is None or chunk == 'done':
                    continue
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices',[]):
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
                    delta = choice.get('delta',{}).get('content')
                    if delta:
                        now = time.perf_counter()-started
                        first = now if first is None else first
                        event_times.append(now)
                        content += delta
        row.update(status='returned',ttft_seconds=first,engine_e2e_seconds=time.perf_counter()-started,
                   content_event_gaps_seconds=[b-a for a,b in zip(event_times,event_times[1:])],
                   event_gap_not_token_itl=True,usage=usage,finish_reason=finish,content=content)
        row['quality']=check_answer(content,item,finish)
        row['validated_elapsed_seconds']=time.perf_counter()-started
        row['original_35s_provider_budget_pass']=row['engine_e2e_seconds'] <= 35
    except Exception as exc:
        row.update(status='failed',error=type(exc).__name__,elapsed_seconds=time.perf_counter()-started)
    return row


async def benchmark(args):
    import httpx
    value = load_workload(args.workload)
    if args.output.exists():
        raise ValueError('Refusing to overwrite baseline')
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        health = await client.get(args.base_url+'/health')
        health.raise_for_status()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as output:
            for repeat in range(args.repetitions):
                for item in value['cases']:
                    before = await client.get(args.base_url+'/metrics')
                    row = await one(client,args.base_url,value,item,args.label,repeat+1)
                    after = await client.get(args.base_url+'/metrics')
                    row.update(metrics_before=before.text,metrics_after=after.text)
                    output.write(json.dumps(row,ensure_ascii=False)+'\n')
                    output.flush()
                    print(json.dumps({k:row[k] for k in ('case','label','repetition','status')} | {'ttft':row.get('ttft_seconds'),'e2e':row.get('engine_e2e_seconds'),'quality':row.get('quality',{}).get('machine_pass')}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    cap = sub.add_parser('capture'); cap.add_argument('--output',type=Path,required=True)
    dl = sub.add_parser('download'); dl.add_argument('--directory',type=Path,required=True); dl.add_argument('--proxy'); dl.add_argument('--weights',action='store_true')
    native = sub.add_parser('download-native'); native.add_argument('--directory',type=Path,required=True); native.add_argument('--proxy')
    tok = sub.add_parser('tokenize'); tok.add_argument('--workload',type=Path,required=True); tok.add_argument('--model-directory',type=Path,required=True); tok.add_argument('--output',type=Path,required=True)
    agent = sub.add_parser('inspect-agent'); agent.add_argument('--archive',type=Path,required=True); agent.add_argument('--model-directory',type=Path,required=True); agent.add_argument('--output',type=Path,required=True)
    bench = sub.add_parser('benchmark'); bench.add_argument('--workload',type=Path,required=True); bench.add_argument('--output',type=Path,required=True); bench.add_argument('--base-url',default='http://127.0.0.1:18110'); bench.add_argument('--label',required=True); bench.add_argument('--repetitions',type=int,default=2)
    args = parser.parse_args()
    if args.action=='capture':
        from power_forecast_service.serve import make_event_loop
        asyncio.run(capture(args.output),loop_factory=make_event_loop)
    elif args.action=='download':
        download(args.directory,args.proxy,args.weights)
    elif args.action=='download-native':
        download_native(args.directory,args.proxy)
    elif args.action=='tokenize':
        tokenize(args.workload,args.model_directory,args.output)
    elif args.action=='inspect-agent':
        inspect_agent(args.archive,args.model_directory,args.output)
    else:
        asyncio.run(benchmark(args))


if __name__=='__main__':
    main()
