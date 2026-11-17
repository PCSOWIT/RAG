"""
LoRA 微调脚本：让 Qwen2.5-1.5B-Instruct 学会"照商品资料回答、没资料就兜底"。

核心思路：
  1. 原模型全部参数冻结，只训练挂在旁边的低秩矩阵（LoRA）
  2. 只对 assistant 的回答段回传梯度，用户提问部分不算 loss
  3. 训练完只保存几个 MB 的 adapter，不复制整个模型

用法：
  先设环境变量 HF_ENDPOINT=https://hf-mirror.com
  python train_lora.py
"""
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = r"D:\models\Qwen2.5-1.5B-Instruct"
DATA_FILE = Path(__file__).parent / "data" / "finetune.jsonl"
OUT_DIR = Path(__file__).parent / "lora_out"

MAX_LEN = 512          # 最长序列长度，数据都很短，512 绰绰有余
EPOCHS = 3
BATCH = 1              # 显存只有 8G，单条单条喂
ACCUM = 8              # 攒 8 条算一次梯度，等效 batch=8
LR = 2e-4              # LoRA 常用学习率


def main():
    print(f"CUDA 可用: {torch.cuda.is_available()} | 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ---------- 1. 分词器 ----------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # Qwen 默认没有 pad token，借用结束符

    # ---------- 2. 读数据并转成模型输入 ----------
    records = [json.loads(line) for line in DATA_FILE.open(encoding="utf-8")]
    print(f"原始样本: {len(records)} 条")

    def to_features(rec):
        msgs = rec["messages"]
        # 前两段（system + user）拼成 prompt，只作为上下文，不训练
        prompt_text = tokenizer.apply_chat_template(
            msgs[:2], tokenize=False, add_generation_prompt=True
        )
        # 第三段（assistant）是答案，末尾补结束符，让它学会"说完就停"
        answer_text = msgs[2]["content"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:MAX_LEN]
        # labels 里 -100 表示"这个位置不算 loss"，所以只有答案部分在训练
        labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LEN]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    ds = Dataset.from_list([to_features(r) for r in records])
    print(f"编码完成，样本数: {len(ds)}")
    print(f"样例 input 长度: {len(ds[0]['input_ids'])}，其中参与训练的位置: {sum(1 for x in ds[0]['labels'] if x != -100)}")

    # ---------- 3. 加载模型 ----------
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},          # 全部放到 0 号显卡（只有一张）
        trust_remote_code=True,
    )
    # 梯度检查点：省显存，代价是慢一点
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ---------- 4. 挂 LoRA ----------
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                          # 低秩矩阵的"秩"，越小越省显存
        lora_alpha=16,                # 缩放系数，一般取 r 的 2 倍
        lora_dropout=0.05,
        # 只在这几种线性层上挂旁路，这是注意力与 MLP 的核心投影层
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()   # 会打印"可训练参数 / 总参数"的比例

    # ---------- 5. 训练参数 ----------
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        learning_rate=LR,
        logging_steps=5,
        save_strategy="no",           # 数据量小，不存中间检查点
        bf16=True,
        report_to="none",
        remove_unused_columns=False,  # 必须关掉，否则 Trainer 会删掉我们自定义的列
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,   # transformers 5.x 起，tokenizer 参数改名为 processing_class
    )

    # ---------- 6. 开跑 ----------
    print("\n开始训练...")
    result = trainer.train()
    print(f"\n训练完成，最终 loss: {result.training_loss:.4f}")

    # ---------- 7. 只保存 adapter ----------
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"adapter 已保存到: {OUT_DIR}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
