# 这一步是基于我“按层级检索”的思路，按时次分块
import os
import json
import re
from typing import List, Dict, Any

def extract_valid_time_from_metadata(metadata_content: str) -> str:
    """
    从元数据内容中提取有效时间
    """
    # 尝试多种格式匹配
    time_patterns = [
        r'记录时间[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日\d{1,2}时)',  # 完整格式
        r'记录时间[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日)',  # 无小时格式
        r'(\d{4}年\d{1,2}月\d{1,2}日\d{1,2}时)',  # 直接匹配时间格式
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, metadata_content)
        if match:
            return match.group(1)
    
    return ""

def chunk_by_section(data_dir: str, output_file: str = "all_chunks.json"):
    """
    将每个.md文件按章节切块，并保存到一个大的JSON文件中。
    改进版本：提取有效时间并替换record_time为valid_time
    """
    chunks = []
    md_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]
    
    print(f"找到 {len(md_files)} 个.md文件")
    
    for i, filename in enumerate(md_files, 1):
        filepath = os.path.join(data_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取文件ID
            file_id_match = re.search(r'ID: (\d+)', content)
            file_id = file_id_match.group(1) if file_id_match else f"unknown_{i}"
            
            # 首先从整个文件中提取有效时间
            # 先匹配元数据部分
            metadata_pattern = r'## 1\. 元数据信息(.*?)(?=^## |\Z)'
            metadata_match = re.search(metadata_pattern, content, re.DOTALL | re.MULTILINE)
            
            valid_time = ""
            if metadata_match:
                metadata_content = metadata_match.group(1)
                valid_time = extract_valid_time_from_metadata(metadata_content)
            else:
                # 如果找不到元数据部分，尝试从整个内容中提取
                valid_time = extract_valid_time_from_metadata(content)
            
            # 如果还是找不到，尝试从文件名中提取
            if not valid_time:
                time_from_filename = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日\d{1,2}时)', filename)
                if time_from_filename:
                    valid_time = time_from_filename.group(1)
            
            # 按章节切分
            # 使用正则表达式匹配每个章节
            # 章节格式：## 1. 元数据信息  或者 ## 2. 天气摘要与风险
            pattern = r'## (\d+\..+?)(?=\n## |\Z)'
            section_matches = re.findall(pattern, content, re.DOTALL)
            
            # 如果没有匹配到章节，则将整个文件作为一个块
            if not section_matches:
                chunk = {
                    "chunk_id": f"{file_id}_whole",
                    "file_id": file_id,
                    "filename": filename,
                    "valid_time": valid_time,  # 使用valid_time
                    "section_title": "整个文件",
                    "content": content,
                    "chunk_type": "whole"
                }
                chunks.append(chunk)
            else:
                for section in section_matches:
                    # 提取章节标题和内容
                    lines = section.strip().split('\n', 1)
                    section_title = lines[0].strip()
                    section_content = lines[1] if len(lines) > 1 else ""
                    
                    # 确定块类型
                    if section_title.startswith("1."):
                        chunk_type = "metadata"
                    elif section_title.startswith("2."):
                        chunk_type = "summary"
                    elif section_title.startswith("3."):
                        chunk_type = "core_parameters"
                    elif section_title.startswith("4."):
                        chunk_type = "detailed_data"
                    elif section_title.startswith("5."):
                        chunk_type = "keywords"
                    else:
                        chunk_type = "other"
                    
                    # 为每个章节生成唯一的chunk_id
                    # 使用文件ID和章节类型，但同类型的章节会有多个，需要区分
                    # 这里我们可以用section_title的简单编码
                    section_code = section_title.replace('.', '_').replace(' ', '_')
                    chunk_id = f"{file_id}_{section_code}"
                    
                    chunk = {
                        "chunk_id": chunk_id,
                        "file_id": file_id,
                        "filename": filename,
                        "valid_time": valid_time,  # 使用valid_time
                        "section_title": section_title,
                        "content": f"## {section}",  # 保留章节标记
                        "chunk_type": chunk_type
                    }
                    chunks.append(chunk)
            
            if i % 1000 == 0:
                print(f"已处理 {i}/{len(md_files)} 个文件，生成 {len(chunks)} 个块")
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n切块完成，共生成 {len(chunks)} 个块，保存到 {output_file}")
    
    # 打印一些统计信息
    time_stats = {}
    for chunk in chunks:
        time_key = chunk.get("valid_time", "未知时间")
        if time_key in time_stats:
            time_stats[time_key] += 1
        else:
            time_stats[time_key] = 1
    
    print(f"时间字段统计:")
    print(f"  有效时间数量: {len([t for t in time_stats if t != ''])}")
    print(f"  空时间数量: {time_stats.get('', 0)}")
    
    return chunks

def validate_chunks(chunks_file: str):
    """验证切块结果"""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"总共 {len(chunks)} 个块")
    
    # 检查前10个块
    print("\n前10个块的valid_time字段:")
    for i, chunk in enumerate(chunks[:10]):
        print(f"  {i+1}. chunk_id: {chunk.get('chunk_id')}")
        print(f"     valid_time: {chunk.get('valid_time')}")
        print(f"     section_title: {chunk.get('section_title')}")
        print()
    
    # 检查时间格式统计
    time_formats = {}
    for chunk in chunks:
        time_val = chunk.get("valid_time", "")
        if time_val:
            # 简单分类
            if "时" in time_val:
                time_formats["完整格式(含小时)"] = time_formats.get("完整格式(含小时)", 0) + 1
            elif "日" in time_val:
                time_formats["日期格式(不含小时)"] = time_formats.get("日期格式(不含小时)", 0) + 1
            else:
                time_formats["其他格式"] = time_formats.get("其他格式", 0) + 1
        else:
            time_formats["空时间"] = time_formats.get("空时间", 0) + 1
    
    print("时间格式统计:")
    for fmt, count in time_formats.items():
        print(f"  {fmt}: {count}")
    
    return chunks

if __name__ == "__main__":
    data_dir = "/home/guojn/code/qikeyuan/DATA/processed_meteorological_data/3-data"
    output_file = "/home/guojn/code/qikeyuan/DATA/processed_meteorological_data/all_chunks.json"
    
    # 执行切块
    chunks = chunk_by_section(data_dir, output_file)
#顺利的话，会得到一个json文件，也就是rag_data路径下的all_chunks.json
