import argparse
import sys
from pathlib import Path
from embeding_pipeline import MeteorologyEmbedder
import os
os.environ["cuda_VISIBLE_DEVICES"]='1' 
def parse_arguments():
    parser = argparse.ArgumentParser(description="气象数据向量嵌入")
    
    parser.add_argument(
        "--chunks_file", 
        type=str, 
        required=True,
        help="切块JSON文件路径"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="BAAI/bge-large-zh-v1.5",
        help="嵌入模型名称，可以是本地路径或HuggingFace模型名"
    )
    
    parser.add_argument(
        "--db_dir", 
        type=str, 
        default="./chroma_meteorology_db",
        help="向量数据库存储目录"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=256,
        help="批处理大小，根据内存调整"
    )
    
    parser.add_argument(
        "--test_query", 
        type=str, 
        default="夏季高对流天气",
        help="测试查询文本"
    )
    
    parser.add_argument(
        "--skip_embedding", 
        action="store_true",
        help="跳过嵌入，只加载现有数据库"
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 检查输入文件
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        print(f"错误: 文件不存在: {chunks_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("气象数据向量嵌入系统")
    print("=" * 60)
    print(f"切块文件: {chunks_file}")
    print(f"嵌入模型: {args.model_name}")
    print(f"数据库目录: {args.db_dir}")
    print(f"批处理大小: {args.batch_size}")
    print("=" * 60)
    
    try:
        # 初始化嵌入器
        embedder = MeteorologyEmbedder(
            model_name=args.model_name,
            persist_dir=args.db_dir,
            batch_size=args.batch_size
        )
        
        if not args.skip_embedding:
            # 1. 加载文本块
            chunks = embedder.load_chunks(str(chunks_file))
            
            # 2. 执行嵌入
            result = embedder.embed_all_chunks(chunks)
            
            print("\n" + "=" * 60)
            print("嵌入完成统计:")
            print(f"  总块数: {result['total']}")
            print(f"  已存在: {result['existing']}")
            print(f"  新块数: {result['new']}")
            print(f"  成功嵌入: {result['success']}")
            print(f"  失败批次: {result['failed']}")
            
            if result['failed'] > 0:
                print(f"\n失败的批次:")
                for fb in result['failed_batches']:
                    print(f"  批次 {fb['batch_number']}: {fb['error']}")
        else:
            print("跳过嵌入步骤，使用现有数据库")
        
        # 3. 获取统计信息
        stats = embedder.get_collection_stats()
        print(f"\n向量数据库统计:")
        print(f"  总向量数: {stats.get('total_chunks', 'N/A')}")
        
        # 4. 测试查询
        if args.test_query:
            print(f"\n测试查询: '{args.test_query}'")
            embedder.query_test(args.test_query, n_results=3)
        
        print("\n" + "=" * 60)
        print("嵌入流程完成！")
        print(f"向量数据库已保存到: {args.db_dir}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
# 顺利的话，该目录下会生成一个向量数据库.xxx_db