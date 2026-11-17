"""
训前 / 训后对比：同一批问题，分别用原始模型和挂上 LoRA adapter 的模型回答。
两边都用贪心解码（do_sample=False），保证结果可复现、可对比。
"""
import gc
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\models\Qwen2.5-1.5B-Instruct"
ADAPTER = r"D:\PythonProject\lora_out"

SYSTEM = (
    "你是数码电子产品店铺的在线客服，主营手机、笔记本、耳机、平板。"
    "像真人客服一样回答：准确、简洁、口语化。"
    "只依据【商品资料】回答价格、配置、库存、活动；"
    "资料里没有的一律回复：这款的信息我这边暂时查不到，建议咨询人工客服确认。"
)

# 覆盖三类：训练集内的商品问题 / 训练集内没有的商品 / 明显的负样本
QUESTIONS = [
    "索尼WH-1000XM5多少钱",
    "AirPods Pro 2什么配置",
    "华为Mate 70 多少钱",
    "你们食堂几点开门",
]


def ask(model, tok, question):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,          # 贪心解码，两次跑结果一致
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    # ---------- 第一轮：原始模型 ----------
    print("=" * 60)
    print("【训前】原始 Qwen2.5-1.5B-Instruct")
    print("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    base_answers = {}
    for q in QUESTIONS:
        ans = ask(model, tok, q)
        base_answers[q] = ans
        print(f"\n问：{q}\n答：{ans}")

    # 释放显存，否则加载第二个模型会爆
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n显存已释放，当前占用: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # ---------- 第二轮：原始模型 + LoRA ----------
    print("\n" + "=" * 60)
    print("【训后】Qwen2.5-1.5B-Instruct + LoRA adapter")
    print("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    tuned_answers = {}
    for q in QUESTIONS:
        ans = ask(model, tok, q)
        tuned_answers[q] = ans
        print(f"\n问：{q}\n答：{ans}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print("【对比汇总】")
    print("=" * 60)
    for q in QUESTIONS:
        same = "相同" if base_answers[q] == tuned_answers[q] else "有变化"
        print(f"\n问题: {q}  [{same}]")
        print(f"  训前: {base_answers[q][:80]}")
        print(f"  训后: {tuned_answers[q][:80]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
