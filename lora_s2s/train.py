import json
import torch
import random
import numpy as np
from typing import Dict, List
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# 设置随机种子
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_data(data_path: str) -> DatasetDict:
    """
    加载训练数据
    
    参数:
        data_path: JSON文件路径
        
    返回:
        DatasetDict对象，包含训练集和验证集
    """
    print(f"加载训练数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 打乱数据
    random.shuffle(data)
    
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    # 转换为HuggingFace Dataset格式
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

def preprocess_function(examples: Dict, tokenizer, max_length: int = 128) -> Dict:
    """
    数据预处理函数
    
    参数:
        examples: 包含"original"和"rewritten"字段的样本
        tokenizer: 分词器
        max_length: 最大序列长度
        
    返回:
        编码后的模型输入
    """
    # 添加任务前缀
    inputs = [f"重写查询: {q}" for q in examples["original"]]
    targets = examples["rewritten"]
    
    # 编码输入
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    # 编码标签
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    # 将标签的填充标记替换为-100，以便在损失计算时忽略
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_ex] 
        for labels_ex in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # 配置参数
    config = {
        # 数据配置
        "data_path": "/root/lora_s2s/query_rewrite_data.json",  # 您的JSON文件路径
        # 模型配置
        "model_name": "/root/lora_s2s/t5-small-chinese-cluecorpussmall",  # 本地模型路径
        "output_dir": "./lora_query_rewriter",  # 输出目录
        
        # LoRA配置
        "lora_r": 8,           # LoRA秩
        "lora_alpha": 32,      # LoRA alpha参数
        "lora_dropout": 0.1,   # LoRA dropout率
        
        # 训练参数
        "num_train_epochs": 60,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-4,
        "warmup_steps": 100,
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 100,
        "max_length": 64,      # 最大序列长度
        
        # 系统参数
        "fp16": torch.cuda.is_available(),  # 使用半精度训练（如果有GPU）
        "seed": 42,
    }
    
    print("=" * 60)
    print("LoRA微调配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 1. 加载数据
    dataset = load_data(config["data_path"])
    
    # 2. 加载分词器和模型
    print(f"\n加载模型和分词器: {config['model_name']}")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    
    # 3. 配置LoRA
    print("配置LoRA参数...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q", "v"],  # 在query和value投影上应用LoRA
        bias="none"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 预处理数据
    print("\n预处理数据...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 5. 配置数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 6. 配置训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        fp16=config["fp16"],
        seed=config["seed"],
        report_to="none",  # 不连接wandb等
        push_to_hub=False
    )
    
    # 7. 创建训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 8. 开始训练
    print("\n开始训练...")
    train_result = trainer.train()
    
    # 9. 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    
    # 保存训练参数
    with open(f"{config['output_dir']}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n训练完成！模型保存到: {config['output_dir']}")
    
    # 10. 评估模型
    print("\n评估模型...")
    eval_results = trainer.evaluate()
    print(f"验证集损失: {eval_results['eval_loss']:.4f}")
    
    # 11. 测试推理
    print("\n测试推理示例:")
    test_queries = [
        "今天的平均气温是多少？",
        "最大对流有效位能多少？",
        "湿度和降水怎么样？"
    ]
    
    model.eval()
    device = model.device
    for query in test_queries:
        input_text = f"重写查询: {query}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=64,return_token_type_ids=False)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config["max_length"],
                num_beams=3,
                do_sample=False
            )
        
        rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"原始查询: {query}")
        print(f"重写查询: {rewritten}")
        print("-" * 40)

if __name__ == "__main__":
    main()