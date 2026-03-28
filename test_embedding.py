import sys
sys.path.append('.')
from retriever import HierarchicalMeteorologyRetriever

def test_hierarchical_retrieval():
    # 初始化检索器
    retriever = HierarchicalMeteorologyRetriever(
        model_path="/root/rag/RAG_data/embeding_model/bge-large-zh-v1.5",
        vector_db_path="/root/rag/RAG_data/.my_vector_db",
        collection_name: str = "meteorology_chunks",
        use_rewrite: bool = False,
        rewrite_model_path: str = "/root/lora_s2s/t5-small-chinese-cluecorpussmall"
    )
    # 测试查询
    queries = [
        "2001年06月03日03时,核心气象参数。平均气温和最高气温分别是多少？",
         "2002年07月15日12时,摘要。总体上来说，气温如何？",
         "1991年08月27日21时,细节。总体上来说，详细气象数据。这一天的最大对流有效位能是多少？"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"测试查询: {query}")
        
        result = retriever.retrieve(query, top_k=3)
        
        if result["success"] and result["results"]:
            print(f"检索到 {len(result['results'])} 个相关记录")
            
            for i, r in enumerate(result["results"], 1):
                metadata = r["metadata"]
                print(f"\n记录 {i}:")
                print(f"  文件ID: {metadata.get('file_id')}")
                print(f"  时间: {metadata.get('valid_time')}")
                print(f"  章节: {metadata.get('section_title')}")
                print(f"  相似度: {r.get('similarity', 0):.4f}")
                
                if r.get("extracted_answer"):
                    print(f"  ✅ 直接提取答案: {r['extracted_answer']}")
        else:
            print("没有检索到相关记录")

if __name__ == "__main__":
    test_hierarchical_retrieval()

# # 检查数据库中是否存在这些时间点的数据
# from retriever import HierarchicalMeteorologyRetriever

# retriever = HierarchicalMeteorologyRetriever(
#     model_path="/root/rag/RAG_data/embeding_model/bge-large-zh-v1.5",
#     vector_db_path="/root/rag/RAG_data/.my_vector_db"
# )

# # 直接查询数据库，不进行向量相似度搜索
# test_times = [
#     "2001年06月03日03时", 
#     "2002年07月15日12时",  
#     "1991年08月27日21时"   
# ]

# for time_str in test_times:
#     print(f"\n检查时间: {time_str}")
#     try:
#         # 仅按时间过滤
#         results = retriever.collection.get(
#             where={"valid_time": {"$eq": time_str}},
#             limit=5
#         )
#         print(f"  找到 {len(results['ids'])} 个记录")
#         if results['ids']:
#             for i, metadata in enumerate(results['metadatas'][:2]):
#                 print(f"    记录{i+1}: {metadata.get('section_title', 'N/A')}")
#     except Exception as e:
#         print(f"  查询错误: {e}")