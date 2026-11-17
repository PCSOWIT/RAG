"""
整理微调数据集：把知识库里的商品资料，转成问答对（JSONL）。
不训练模型，只做数据准备——微调本身需要 GPU，这里只交付可直接喂给训练框架的数据。

输出格式是 messages 格式（OpenAI / LLaMA-Factory / ms-swift 都认）：
{"messages": [{"role": "system"...}, {"role": "user"...}, {"role": "assistant"...}]}

关键设计：负样本混在里面。正样本教它"怎么答"，负样本教它"什么时候闭嘴"。
只喂正样本，微调出来的模型会更敢编。
"""
import json
import sys
from pathlib import Path

import psycopg

DSN = "host=127.0.0.1 port=5433 dbname=ai_kb user=postgres"
OUT = Path(__file__).parent.parent / "data" / "finetune.jsonl"

SYSTEM = (
    "你是数码电子产品店铺的在线客服，主营手机、笔记本、耳机、平板。"
    "像真人客服一样回答：准确、简洁、口语化。"
    "只依据【商品资料】回答价格、配置、库存、活动；"
    "资料里没有的一律回复：这款的信息我这边暂时查不到，建议咨询人工客服确认。"
)

FALLBACK = "这款的信息我这边暂时查不到，建议咨询人工客服确认。"

# 负样本：训练模型"识别不该回答的问题"
NEG_QUESTIONS = [
    "你们食堂几点开门",
    "今天天气怎么样",
    "帮我写一首诗",
    "明天股票会涨吗",
    "你叫什么名字",
]


def parse_item(content: str):
    """把 'iPhone 16，售价5999元，128G存储，A18芯片' 拆成 (名称, 价格, 其他属性)。"""
    parts = [p.strip() for p in content.replace("\n", "，").split("，") if p.strip()]
    if len(parts) < 2 or len(parts[0]) > 30:
        return None
    name = parts[0]
    price = None
    feats = []
    for p in parts[1:]:
        if "售价" in p or "学费" in p:
            price = p
        else:
            feats.append(p)
    return name, price, feats


def main():
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content, category FROM kb_chunks ORDER BY id")
            rows = cur.fetchall()

    samples = []
    for content, cat in rows:
        parsed = parse_item(content)
        if not parsed:
            continue
        name, price, feats = parsed

        # ① 问价格 —— 三种问法，让模型学会不同表述下都给同一个数
        if price:
            answer = price + "。" + ("，".join(feats) if feats else "")
            for q in (f"{name}多少钱", f"{name}什么价格", f"{name}卖多少钱"):
                samples.append((q, answer, "price"))

        # ② 问配置 —— 价格可以不提，避免每次都硬塞数字
        if feats:
            answer = "，".join(feats) + "。"
            for q in (f"{name}什么配置", f"{name}有什么参数"):
                samples.append((q, answer, "spec"))

        # ③ 开放问法
        samples.append((f"介绍一下{name}", content, "intro"))

    # ④ 负样本
    for q in NEG_QUESTIONS:
        samples.append((q, FALLBACK, "negative"))

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for q, a, tag in samples:
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "meta": {"type": tag},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    stats = Counter(t for _, _, t in samples)
    print(f"生成 {len(samples)} 条 -> {OUT.relative_to(OUT.parent.parent)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
