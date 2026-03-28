import os
import re
import json
import chromadb
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HierarchicalMeteorologyRetriever:
    """分层气象数据检索器"""
    
    def __init__(self, 
                 model_path: str,
                 vector_db_path: str = "/root/rag/RAG_data/.my_vector_db",
                 collection_name: str = "meteorology_chunks",
                 use_rewrite: bool = True,
                 rewrite_model_path: str = "/root/lora_s2s/t5-small-chinese-cluecorpussmall"):
        """
        初始化检索器
        
        参数:
            model_path: 本地嵌入模型路径
            vector_db_path: 向量数据库路径
            collection_name: 集合名称
            use_rewrite: 是否使用查询重写
            rewrite_model_path: 查询重写模型路径
        """
        # 设置离线模式
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # 加载嵌入模型
        logger.info(f"加载嵌入模型: {model_path}")
        self.model = SentenceTransformer(model_path)
        
        # 连接向量数据库
        logger.info(f"连接向量数据库: {vector_db_path}")
        self.client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.client.get_collection(name=collection_name)

        self.use_rewrite = use_rewrite
        if use_rewrite:
            try:
                from inference_lora import LoRAQueryRewriter
                self.query_rewriter = LoRAQueryRewriter(rewrite_model_path) 
                logger.info(f"查询重写器加载成功: {rewrite_model_path}")
            except Exception as e:
                logger.error(f"查询重写器加载失败: {e}")
                logger.warning("将禁用查询重写功能")
                self.use_rewrite = False
        
        # 章节映射表
        self.section_mapping = {
            "元数据信息": ["metadata"],
            "元数据": ["metadata"],
            
            "天气摘要与风险": ["summary"],
            "天气摘要": ["summary"],
            "摘要": ["summary"],
            "风险": ["summary"],
            "总览":["summary"],
            "大致":["summary"],
            "大概":["summary"],
            "基本信息": ["summary"],
            
            "核心气象参数": ["core_parameters"],
            "核心参数": ["core_parameters"],
            "主要参数": ["core_parameters"],
            "主要": ["core_parameters"],
            "重要参数": ["core_parameters"],
            
            "详细气象数据": ["detailed_data"],
            "详细数据": ["detailed_data"],
            "具体数据": ["detailed_data"],
            "详细": ["detailed_data"],
            "专业": ["detailed_data"],
            
            "检索关键词": ["keywords"],
            "关键词": ["keywords"],
        }
        
        logger.info("检索器初始化完成")
    
    def parse_query(self, query: str) -> Dict[str, str]:
        """解析用户查询，格式：[时间]，[章节]。具体问题"""
        # 使用正则表达式匹配格式
        pattern = r'(.+?)[，,]\s*(.+?)[。.]\s*(.+)'
        match = re.match(pattern, query)
        
        if not match:
            raise ValueError(
                "查询格式不正确！请使用格式：[时间]，[章节]。具体问题\n"
                "例如：2023年06月03日03时，详细气象数据。这一天的最大对流有效位能是多少？"
            )
        
        time_part = match.group(1).strip()
        section_part = match.group(2).strip()
        question_part = match.group(3).strip()
        
        return {
            "time": time_part,
            "section": section_part,
            "question": question_part
        }
    
    def hierarchical_retrieve(self, 
                            time: str, 
                            section: str, 
                            question: str, 
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """分层检索"""
        logger.info(f"开始分层检索: 时间={time}, 章节={section}, 问题={question}")
        
        # 1. 映射章节到chunk_type
        chunk_types = self.map_section_to_chunk_types(section)
        logger.info(f"章节 '{section}' 映射到 chunk_types: {chunk_types}")
        
        # 2. 构建元数据过滤条件
        where_filter = {"$and": []}
        
        if time:
            where_filter["$and"].append({"valid_time": {"$eq": time}})
        
        if chunk_types:
            if len(chunk_types) == 1:
                where_filter["$and"].append({"chunk_type": {"$eq": chunk_types[0]}})
            else:
                where_filter["$and"].append({"chunk_type": {"$in": chunk_types}})
        
        # 3. 生成问题向量
        question_embedding = self.model.encode(
            question, 
            normalize_embeddings=True
        ).tolist()
        
        # 4. 执行分层检索
        try:
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=top_k * 3,
                where=where_filter
            )
            
            # 5. 处理结果
            processed_results = []
            if results and results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    distance = results['distances'][0][i] if results.get('distances') else None
                    similarity = 1 - distance if distance is not None else None
                    
                    processed_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity": similarity,
                        "rank": i + 1
                    })
                
                # 按相似度排序
                processed_results.sort(key=lambda x: x["similarity"] or 0, reverse=True)
                processed_results = processed_results[:top_k]
            
            return processed_results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def map_section_to_chunk_types(self, user_section: str) -> List[str]:
        """将用户输入的章节映射到数据库中的chunk_type"""
        # 尝试精确匹配
        for key, chunk_types in self.section_mapping.items():
            if key == user_section:
                return chunk_types
        
        # 尝试部分匹配
        for key, chunk_types in self.section_mapping.items():
            if key in user_section or user_section in key:
                return chunk_types
        
        # 默认summary类型
        return ["summary"]
    
    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        完整的检索流程
        
        参数:
            query: 用户查询字符串
            top_k: 返回结果数量
            
        返回:
            包含检索结果的字典
        """
        try:
            # 1. 解析查询
            parsed = self.parse_query(query)
            logger.info(f"查询解析结果: {parsed}")
            
            # 获取原始问题
            original_question = parsed["question"]
            
            # 2. 查询重写
            question_for_retrieval = original_question  # 默认使用原始问题
            
            if self.use_rewrite and hasattr(self, 'query_rewriter'):
                try:
                    rewritten_question = self.query_rewriter.rewrite(original_question)
                    logger.info(f"[查询重写] 原始: '{original_question}'")
                    logger.info(f"[查询重写] 重写: '{rewritten_question}'")
                    question_for_retrieval = rewritten_question
                except Exception as e:
                    logger.warning(f"[查询重写失败] {e}")
                    # 重写失败，继续使用原始问题
            # 3. 分层检索
            results = self.hierarchical_retrieve(
                time=parsed["time"],
                section=parsed["section"],
                question=question_for_retrieval,
                top_k=top_k
            )
            
            # 4. 构建返回结果
            response = {
                "success": True,
                "query": query,
                "parsed": parsed,
                "original_question": original_question,
                "rewritten_question": question_for_retrieval,
                "results_count": len(results),
                "results": results
            }
            
            return response
            
        except Exception as e:
            logger.error(f"检索过程出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }