# 这里使用了bge-large-zh-v1.5做嵌入模型
import json
import time
import hashlib
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
#  %%
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MeteorologyEmbedder:
    """气象数据嵌入器"""
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-large-zh-v1.5",
        persist_dir: str = "./chroma_meteorology_db",
        batch_size: int = 256,
        max_retries: int = 3
    ):
        """
        初始化嵌入器
        
        参数:
            model_name: 嵌入模型名称或路径
            persist_dir: ChromaDB持久化目录
            batch_size: 批处理大小
            max_retries: 最大重试次数
        """
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # 创建存储目录
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 初始化向量数据库
        self._init_vector_db()
    
    def _init_model(self):
        """初始化嵌入模型"""
        logger.info(f"加载嵌入模型: {self.model_name}")
        try:
            # 尝试从本地路径加载
            if Path(self.model_name).exists():
                self.model = SentenceTransformer(self.model_name)
            else:
                # 从HuggingFace下载
                self.model = SentenceTransformer(
                    self.model_name, 
                    device='cpu'  # 如果有GPU可以改为'cuda'
                )
            logger.info(f"模型加载成功，向量维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _init_vector_db(self):
        """初始化向量数据库"""
        logger.info(f"初始化ChromaDB，存储路径: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # 创建或获取集合
        try:
            self.collection = self.client.get_collection(name="meteorology_chunks")
            logger.info("使用现有集合: meteorology_chunks")
        except:
            self.collection = self.client.create_collection(
                name="meteorology_chunks",
                metadata={"description": "气象数据分析记录数据库"},
                embedding_function=None  # 我们手动提供嵌入向量
            )
            logger.info("创建新集合: meteorology_chunks")
    
    def load_chunks(self, json_file: str) -> List[Dict]:
        """从JSON文件加载文本块"""
        logger.info(f"加载JSON文件: {json_file}")
        
        if not Path(json_file).exists():
            raise FileNotFoundError(f"文件不存在: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"成功加载 {len(chunks)} 个文本块")
        
        # 数据验证
        validated_chunks = self._validate_chunks(chunks)
        
        return validated_chunks
    
    def _validate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """验证和清洗文本块"""
        logger.info("验证和清洗文本块...")
        
        validated = []
        empty_content_count = 0
        duplicate_ids = set()
        
        for i, chunk in enumerate(chunks):
            # 确保有唯一的chunk_id
            if "chunk_id" not in chunk or not chunk["chunk_id"]:
                # 使用hash生成唯一的ID
                content_hash = hashlib.md5(chunk.get("content", "").encode()).hexdigest()[:16]
                chunk["chunk_id"] = f"chunk_{content_hash}"
            
            # 检查重复ID
            if chunk["chunk_id"] in duplicate_ids:
                logger.warning(f"发现重复的chunk_id: {chunk['chunk_id']}")
                # 添加时间戳避免重复
                chunk["chunk_id"] = f"{chunk['chunk_id']}_{int(time.time())}"
            
            duplicate_ids.add(chunk["chunk_id"])
            
            # 检查内容是否为空
            content = chunk.get("content", "").strip()
            if not content or len(content) < 10:
                empty_content_count += 1
                logger.warning(f"文本块 {chunk['chunk_id']} 内容过短: {len(content)} 字符")
                continue
            
            validated.append(chunk)
        
        logger.info(f"验证完成: 有效块 {len(validated)}，跳过 {empty_content_count} 个空内容块")
        return validated
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """嵌入一个批次的文本"""
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,  # 归一化向量，有利于相似度计算
                batch_size=min(32, len(texts))  # 模型内部批处理大小
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"嵌入批处理失败: {e}")
            raise
    
    def embed_all_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """嵌入所有文本块"""
        logger.info("开始嵌入处理...")
        
        total_chunks = len(chunks)
        logger.info(f"总共 {total_chunks} 个文本块，批处理大小: {self.batch_size}")
        
        # 检查哪些块已经存在
        existing_ids = self._get_existing_ids()
        new_chunks = []
        
        for chunk in chunks:
            if chunk["chunk_id"] not in existing_ids:
                new_chunks.append(chunk)
        
        logger.info(f"已存在 {len(existing_ids)} 个块，需要嵌入 {len(new_chunks)} 个新块")
        
        if not new_chunks:
            logger.info("没有新块需要嵌入")
            return {"total": 0, "success": 0, "failed": 0}
        
        # 分批处理
        success_count = 0
        failed_batches = []
        
        num_batches = (len(new_chunks) - 1) // self.batch_size + 1
        
        with tqdm(total=num_batches, desc="嵌入进度") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(new_chunks))
                batch = new_chunks[start_idx:end_idx]
                
                batch_number = batch_idx + 1
                logger.info(f"处理批次 {batch_number}/{num_batches}: 块 {start_idx}-{end_idx}")
                
                try:
                    # 准备批次数据
                    batch_ids = [chunk["chunk_id"] for chunk in batch]
                    batch_documents = [chunk["content"] for chunk in batch]
                    batch_metadatas = self._prepare_metadata(batch)
                    
                    # 生成嵌入向量
                    logger.info(f"为批次 {batch_number} 生成嵌入向量...")
                    batch_embeddings = self.embed_batch(batch_documents)
                    
                    # 添加到向量数据库
                    self.collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    
                    success_count += len(batch)
                    logger.info(f"批次 {batch_number} 处理成功: {len(batch)} 个块")
                    
                except Exception as e:
                    logger.error(f"批次 {batch_number} 处理失败: {e}")
                    failed_batches.append({
                        "batch_number": batch_number,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "error": str(e)
                    })
                    
                    # 保存失败的批次
                    self._save_failed_batch(batch, batch_number)
                
                pbar.update(1)
                
                # 每处理10个批次，记录一次进度
                if batch_number % 10 == 0:
                    logger.info(f"进度: 已处理 {end_idx}/{len(new_chunks)} 个新块 ({end_idx/len(new_chunks)*100:.1f}%)")
        
        # 记录结果
        result = {
            "total": total_chunks,
            "existing": len(existing_ids),
            "new": len(new_chunks),
            "success": success_count,
            "failed": len(failed_batches),
            "failed_batches": failed_batches
        }
        
        # 保存统计信息
        self._save_statistics(result)
        
        return result
    
    def _get_existing_ids(self) -> set:
        """获取已存在的向量ID"""
        existing_ids = set()
        try:
            # 分页获取所有ID
            limit = 10000
            offset = 0
            
            while True:
                results = self.collection.get(limit=limit, offset=offset)
                if not results["ids"]:
                    break
                
                existing_ids.update(results["ids"])
                offset += limit
                
                if len(results["ids"]) < limit:
                    break
        except Exception as e:
            logger.warning(f"获取已存在ID失败: {e}")
        
        return existing_ids
    
    def _prepare_metadata(self, batch: List[Dict]) -> List[Dict]:
        """准备元数据"""
        metadatas = []
        
        for chunk in batch:
            metadata = {
                "file_id": chunk.get("file_id", ""),
                "filename": chunk.get("filename", ""),
                "valid_time": chunk.get("valid_time", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "content_length": len(chunk.get("content", "")),
                "embedding_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            metadatas.append(metadata)
        
        return metadatas
    
    def _save_failed_batch(self, batch: List[Dict], batch_number: int):
        """保存失败的批次"""
        failed_dir = Path(self.persist_dir) / "failed_batches"
        failed_dir.mkdir(exist_ok=True)
        
        filepath = failed_dir / f"batch_{batch_number}_{int(time.time())}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        
        logger.info(f"失败的批次已保存到: {filepath}")
    
    def _save_statistics(self, stats: Dict):
        """保存统计信息"""
        stats_file = Path(self.persist_dir) / "embedding_stats.json"
        
        # 如果已存在，追加记录
        existing_stats = []
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                existing_stats = json.load(f)
        
        stats_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": stats
        }
        
        existing_stats.append(stats_record)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(existing_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计信息已保存到: {stats_file}")
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        try:
            count = self.collection.count()
            logger.info(f"向量数据库中的块数量: {count}")
            
            # 获取元数据统计
            sample = self.collection.get(limit=1)
            if sample["metadatas"]:
                metadata_keys = list(sample["metadatas"][0].keys())
                logger.info(f"元数据字段: {metadata_keys}")
            
            return {
                "total_chunks": count,
                "collection_name": "meteorology_chunks"
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {}
    
    def query_test(self, query_text: str, n_results: int = 3):
        """测试查询"""
        logger.info(f"测试查询: '{query_text}'")
        
        try:
            # 生成查询向量
            query_embedding = self.model.encode(
                query_text,
                normalize_embeddings=True
            ).tolist()
            
            # 执行查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            if results and results['documents']:
                logger.info(f"查询成功，返回 {len(results['documents'][0])} 个结果")
                
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    distance = results['distances'][0][i] if results.get('distances') else None
                    similarity = 1 - distance if distance is not None else None
                    
                    print(f"\n结果 {i+1}:")
                    print(f"  相似度: {similarity:.4f}")
                    print(f"  文件ID: {metadata.get('file_id', 'N/A')}")
                    print(f"  时间: {metadata.get('valid_time', 'N/A')}")
                    print(f"  章节: {metadata.get('section_title', 'N/A')}")
                    print(f"  预览: {doc[:150]}...")
                    
                    logger.info(f"结果 {i+1}: 相似度={similarity:.4f}, 时间={metadata.get('valid_time')}")
            
            return results
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return None