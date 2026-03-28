import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import json

class LoRAQueryRewriter:
    def __init__(self, model_path: str, device: str = None):
        """
        初始化查询重写器
        
        参数:
            model_path: LoRA模型路径
            device: 运行设备
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载配置
        print(f"加载LoRA模型: {model_path}")
        config = PeftConfig.from_pretrained(model_path)
        
        # 加载基础模型
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path
        )
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path
        )
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(
            base_model, 
            model_path
        )
        
        # 移动到设备
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成")
        
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                print(f"找到LoRA参数: {name}, 可训练: {param.requires_grad}")
        # print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        # print(f"总参数: {total_params:,}")
    
    def rewrite(self, query: str, max_length: int = 50) -> str:
        """
        重写查询
        
        参数:
            query: 原始查询
            max_length: 生成的最大长度
            
        返回:
            重写后的查询
        """
        # 添加任务前缀
        input_text = f"重写查询: {query}"
        
        # 编码
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=64,return_token_type_ids=False)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                temperature=0.7,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # 解码
        rewritten = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        return rewritten
    
    def batch_rewrite(self, queries: list, max_length: int = 50) -> list:
        """批量重写查询"""
        results = []
        for query in queries:
            rewritten = self.rewrite(query, max_length=max_length)
            results.append(rewritten)
        return results

def main():
    # 加载模型
    model_path = "./lora_query_rewriter"  # 训练输出目录
    rewriter = LoRAQueryRewriter(model_path)
    
    # 测试查询
    test_queries = [
        "今天的平均气温是多少？",
        "最大对流有效位能多少？",
        "湿度和降水怎么样？",
        "风场和垂直运动如何？",
        "对流和降水情况如何？",
        "温度和对流能量多少？"
    ]
    
    print("\n测试查询重写:")
    print("=" * 60)
    
    for query in test_queries:
        rewritten = rewriter.rewrite(query)
        print(f"原始查询: {query}")
        print(f"重写查询: {rewritten}")
        print("-" * 40)

if __name__ == "__main__":
    main()