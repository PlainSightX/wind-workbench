"""LoRA SFT：只在 completion（规划JSON + EOS）上计算损失，按 epoch 打乱，余弦学习率。"""
import argparse
import gzip
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def causal_flash(module, query, key, value, attention_mask, **kwargs):
    """单样本无padding：丢弃显式mask，改用is_causal=True，使SDPA走flash内核（反向快约百倍）。"""
    kwargs.pop("is_causal", None)
    return sdpa_attention_forward(module, query, key, value, None, is_causal=True, **kwargs)


AttentionInterface.register("causal_flash", causal_flash)


def encode(tok, messages, completion):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok.encode(prompt, add_special_tokens=False)
    full = tok.encode(prompt + completion, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
    assert full[:len(p)] == p, "prompt_prefix_mismatch"
    labels = [-100] * len(p) + full[len(p):]
    return full, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8, help="每次优化器更新的样本数（梯度累积）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = json.load(gzip.open(args.data, "rt", encoding="utf-8"))
    data = [encode(tok, r["messages"], r["completion"]) for r in rows]
    print("examples", len(data), "max_len", max(len(x[0]) for x in data), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="causal_flash").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(data) / args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    order = []
    while len(order) < total * args.batch:
        idx = list(range(len(data))); random.shuffle(idx); order += idx
    model.train()
    t0 = time.time()
    log = open(out / "train_log.jsonl", "w")
    for step in range(total):
        batch = order[step * args.batch:(step + 1) * args.batch]
        ntok = sum(sum(1 for t in data[i][1] if t != -100) for i in batch)
        loss_sum = 0.0
        for i in batch:
            ids, labels = data[i]
            # completion 在序列末尾：只为最后 m+1 个位置计算 logits，避免整段 23k×词表 的显存
            m = sum(1 for t in labels if t != -100)
            x = torch.tensor([ids], device="cuda")
            target = torch.tensor(labels[-m:], device="cuda")
            logits = model(input_ids=x, logits_to_keep=m + 1).logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits, target, reduction="sum") / ntok
            loss.backward()
            loss_sum += loss.item()
            del logits, loss
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "total": total, "loss": loss_sum, "grad_norm": float(norm),
               "lr": sched.get_last_lr()[0], "elapsed": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % 10 == 0 or step == 0:
            print(rec, flush=True)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    (out / "done.json").write_text(json.dumps({"steps": total, "seconds": time.time() - t0, "args": vars(args)}))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
