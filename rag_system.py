import sys
sys.path.append('.')
from retriever import HierarchicalMeteorologyRetriever
from typing import List, Dict, Any
import re

class RAGQWENSystem:
    def __init__(self, 
                 retriever_config: Dict[str, Any] = None,
                 qwen_model_path: str = None):
        """
        初始化RAG-QWEN系统
        
        参数:
            retriever_config: 检索器配置字典
            qwen_model_path: QWEN模型路径（可选，如果不提供则不加载模型）
        """
        # 默认检索器配置
        default_retriever_config = {
            "model_path": "/root/rag/RAG_data/embeding_model/bge-large-zh-v1.5",
            "vector_db_path": "/root/rag/RAG_data/.my_vector_db",
            "collection_name": "meteorology_chunks",
            "use_rewrite": False,
            "rewrite_model_path": "/root/lora_s2s/t5-small-chinese-cluecorpussmall"
        }
        
        if retriever_config:
            default_retriever_config.update(retriever_config)
        
        # 初始化检索器
        print("初始化检索器...")
        self.retriever = HierarchicalMeteorologyRetriever(**default_retriever_config)
        print("检索器初始化完成")
        
        # 初始化QWEN模型（可选）
        self.model = None
        self.tokenizer = None
        
        if qwen_model_path:
            print(f"加载QWEN模型: {qwen_model_path}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                qwen_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("QWEN模型加载完成")
    
    def clean_content(self, content: str) -> str:
        """
        清理和格式化内容，使其更易读
        
        参数:
            content: 原始内容
            
        返回:
            清理后的内容
        """
        if not content:
            return content
        
        # 1. 移除Markdown标题符号
        content = re.sub(r'#{1,6}\s*', '', content)
        
        # 2. 移除加粗和斜体符号
        content = re.sub(r'[*_]{1,2}', '', content)
        
        # 3. 移除代码块和行内代码符号
        content = re.sub(r'`{1,3}', '', content)
        
        # 4. 移除分割线
        content = re.sub(r'-{3,}', '', content)
        
        # 5. 规范化中文标点
        content = re.sub(r'(\S):\s*', r'\1：', content)  # 将英文冒号替换为中文冒号
        
        
        # 6. 规范化其他单位
        content = re.sub(r'm/s', '米/秒', content)
        content = re.sub(r'J/kg', '焦耳/千克', content)
        content = re.sub(r'Pa/s', '帕/秒', content)
        content = re.sub(r'1/s', '1/秒', content)
        content = re.sub(r'mm', '毫米', content)
        
        # 7. 科学计数法转换
        content = re.sub(r'(\d+\.?\d*)e-0?(\d+)', r'\1×10⁻\2', content)
        
        #8. 移除多余的空白行
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        
        # 9. 移除多余的空白字符
        content = re.sub(r'[ \t]+', ' ', content)
        content = content.strip()
        
        return content
    
    def extract_question_from_query(self, query: str) -> str:
        """
        从完整查询中提取具体问题部分
        
        参数:
            query: 完整查询字符串
            
        返回:
            提取的具体问题
        """
        # 匹配格式：[时间]，[章节]。具体问题
        pattern = r'[，,]\s*[^。，,]*[。.]\s*(.+)'
        match = re.search(pattern, query)
        
        if match:
            return match.group(1).strip()
        
        return query
    
    def build_rag_prompt(self, 
                        user_query: str, 
                        top_k: int = 3,
                        include_system_prompt: bool = True) -> List[Dict[str, str]]:
        """
        构建RAG prompt，返回QWEN格式的消息列表
        
        参数:
            user_query: 用户查询字符串
            top_k: 检索结果数量
            include_system_prompt: 是否包含系统提示
            
        返回:
            QWEN格式的消息列表
        """
        # 1. 检索相关文档
        result = self.retriever.retrieve(user_query, top_k=top_k)
        
        if not result["success"] or not result["results"]:
            # 如果没有检索到结果，返回一个简单的问题
            messages = []
            if include_system_prompt:
                messages.append({
                    "role": "system",
                    "content": "你是一个气象数据分析助手，帮助用户解答气象相关问题。"
                })
            messages.append({
                "role": "user",
                "content": user_query
            })
            return messages
        
        # 2. 提取检索到的内容并清理
        retrieved_contents = []
        for i, r in enumerate(result["results"], 1):
            content = r.get("content", "")
            # 清理和格式化内容
            cleaned_content = self.clean_content(content)
            
            # 可以选择添加元数据信息
            metadata = r.get("metadata", {})
            section = metadata.get("section_title", "未知章节")
            similarity = r.get("similarity", 0)
            
            # 格式化内容，使用更清晰的格式
            if len(cleaned_content) > 300:
                cleaned_content = cleaned_content[:300] + "..."
            
            formatted_content = f"【数据来源 {i}】{section}"
            retrieved_contents.append(formatted_content)
        
        # 3. 合并所有检索内容
        context = "\n\n".join(retrieved_contents)
        
        # 4. 提取具体问题
        question = self.extract_question_from_query(user_query)
        
        # 5. 解析时间信息
        time_match = re.search(r'(\d{4}年\d{2}月\d{2}日\d{2}时)', user_query)
        time_info = f"（观测时间：{time_match.group(1)}）" if time_match else ""
        
        # 6. 构建更清晰易读的prompt
        prompt = f"""以下是一份气象观测数据报告：

{context}

请根据上述气象数据回答我的问题。

我的问题是：{question}{time_info}

回答要求：
1. 请基于数据提供准确的数值，并且说明是来自EAR5的再分析数据
2. 如果数据中没有相关信息，请说明这一点
3. 保持回答的专业性和简洁性"""
        
        # 7. 构建QWEN格式的消息
        messages = []
        
        if include_system_prompt:
            messages.append({
                "role": "system",
                "content": """你是气象数据分析专家，负责解答用户关于气象数据的问题。

请仔细阅读用户提供的气象数据，然后根据数据内容准确、专业地回答用户的问题。

回答原则：
1. 严格基于用户提供的数据进行回答
2. 如果数据中没有相关信息，请诚实说明
3. 回答要简洁明了，避免冗长
4. 使用规范的中文表达"""
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def generate_response(self, 
                         user_query: str, 
                         top_k: int = 3,
                         max_new_tokens: int = 300,
                         temperature: float = 0.7) -> Dict[str, Any]:
        """
        完整的RAG-QWEN流程：检索 -> 构建prompt -> 生成回答
        
        参数:
            user_query: 用户查询
            top_k: 检索结果数量
            max_new_tokens: 生成的最大token数
            temperature: 采样温度
            
        返回:
            包含完整信息的字典
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("QWEN模型未加载，请先提供模型路径")
        
        # 1. 构建prompt
        messages = self.build_rag_prompt(user_query, top_k=top_k)
        
        # 2. 调用QWEN模型生成回答
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with self.model.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 3. 解码回答
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        # 4. 返回完整结果
        return {
            "query": user_query,
            "messages": messages,  # 这是可以重用或检查的
            "response": response,
            "prompt_length": inputs["input_ids"].shape[-1],
            "response_length": len(response)
        }
    
    def get_messages_for_query(self, 
                              user_query: str, 
                              top_k: int = 3) -> List[Dict[str, str]]:
        """
        仅为用户查询生成QWEN格式的消息列表，不调用模型
        
        参数:
            user_query: 用户查询
            top_k: 检索结果数量
            
        返回:
            QWEN格式的消息列表
        """
        return self.build_rag_prompt(user_query, top_k=top_k)

# 使用示例
if __name__ == "__main__":
    rag_system = RAGQWENSystem()
    query = "2001年06月03日03时,总览。平均气温的数值"
    messages = rag_system.get_messages_for_query(query, top_k=2)
    print(messages)