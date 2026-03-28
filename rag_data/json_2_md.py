# 根据json文件，生成对应的markdown文档，确保json是一个列表，即首尾有“[”与"]"
import json
from datetime import datetime
import re
import os

def process_json_list_to_markdown(json_list, output_dir="processed_docs"):
    """
    处理JSON列表，为每个元素生成Markdown文件
    
    参数:
    - json_list: JSON数据列表
    - output_dir: 输出目录
    
    返回:
    - 生成的Markdown文件路径列表
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    
    for i, json_data in enumerate(json_list):
        try:
            # 生成文件名
            if "id" in json_data:
                filename = f"weather_data_{json_data['id']}.md"
            else:
                filename = f"weather_data_{i:05d}.md"
            
            output_path = os.path.join(output_dir, filename)
            
            # 转换为Markdown
            result = json_to_rag_markdown(json_data, output_path)
            
            if result:  # 如果成功生成文件
                output_files.append(output_path)
                
                # 打印进度
                print(f"处理完成: {filename}")
                
        except Exception as e:
            print(f"处理第 {i} 个元素时出错: {e}")
            continue
    
    return output_files

def json_to_rag_markdown(json_data, output_path=None):
    """
    将单个气象JSON数据转换为适合RAG的Markdown文件
    
    参数:
    - json_data: 单个JSON数据字典
    - output_path: 输出Markdown文件路径，如不指定则返回字符串
    
    返回:
    - 如果output_path为None，返回Markdown字符串
    - 否则保存到文件并返回文件路径
    """
    
    # 检查json_data是否为字典
    if not isinstance(json_data, dict):
        raise ValueError("json_data必须是字典类型")
    
    # 检查必要字段
    required_fields = ["metadata", "summary", "detailed_data"]
    for field in required_fields:
        if field not in json_data:
            raise ValueError(f"JSON数据缺少必要字段: {field}")
    
    # 1. 提取元数据
    metadata = extract_metadata(json_data)
    
    # 2. 构建Markdown内容
    md_content = []
    
    # 添加文件头
    time_str = format_datetime(metadata['datetime'])
    if "id" in json_data:
        md_content.append(f"# 气象数据记录 - {time_str} (ID: {json_data['id']})\n")
    else:
        md_content.append(f"# 气象数据记录 - {time_str}\n")
    
    # 第一部分：元数据（作为上下文信息）
    md_content.append("## 1. 元数据信息\n")
    md_content.append(generate_metadata_section(metadata))
    md_content.append("\n---\n")
    
    # 第二部分：摘要总结（最重要的部分）
    md_content.append("## 2. 天气摘要与风险\n")
    md_content.append(generate_summary_section(json_data, metadata))
    md_content.append("\n---\n")
    
    # 第三部分：核心参数分析
    md_content.append("## 3. 核心气象参数\n")
    md_content.append(generate_core_parameters_section(json_data, metadata))
    md_content.append("\n---\n")
    
    # 第四部分：详细数据
    md_content.append("## 4. 详细气象数据\n")
    md_content.append(generate_detailed_section(json_data, metadata))
    md_content.append("\n---\n")
    
    # 第五部分：检索关键词
    md_content.append("## 5. 检索关键词\n")
    md_content.append(generate_keywords_section(metadata, json_data))
    
    # 转换为完整字符串
    full_md = "\n".join(md_content)
    
    # 保存或返回
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_md)
        return output_path
    else:
        return full_md

def extract_metadata(json_data):
    """提取元数据"""
    # 安全地提取元数据，处理可能的缺失字段
    metadata_dict = json_data.get("metadata", {})
    
    return {
        "datetime": metadata_dict.get("datetime", ""),
        "year": metadata_dict.get("year", 0),
        "month": metadata_dict.get("month", 0),
        "day": metadata_dict.get("day", 0),
        "hour": metadata_dict.get("hour", 0),
        "season": metadata_dict.get("season", "unknown"),
        "is_summer_afternoon": metadata_dict.get("is_summer_afternoon", 0.0)
    }

def generate_metadata_section(metadata):
    """生成元数据部分"""
    time_str = format_datetime(metadata["datetime"])
    
    content = [
        f"**记录时间**: {time_str}",
        f"**年份**: {metadata['year']}年",
        f"**月份**: {metadata['month']}月",
        f"**日期**: {metadata['day']}日",
        f"**小时**: {metadata['hour']}时",
        f"**季节**: {metadata['season']}",
        f"**是否为夏季下午**: {'是' if metadata['is_summer_afternoon'] == 1.0 else '否'}"
    ]
    return "\n\n".join(content)

def generate_summary_section(json_data, metadata):
    """生成摘要部分"""
    summary = json_data.get("summary", "")
    
    if not summary:
        return "**无摘要信息**"
    
    # 从摘要中提取风险等级
    risk_level_match = re.search(r"综合天气风险等级：(.+?)。", summary)
    risk_level = risk_level_match.group(1) if risk_level_match else "未知"
    
    # 从摘要中提取主要风险因素
    risk_factors_match = re.search(r"主要风险因素：(.+?)。", summary)
    risk_factors = risk_factors_match.group(1) if risk_factors_match else "未明确"
    
    enhanced_summary = f"""
**风险等级**: {risk_level}

**主要风险因素**: {risk_factors}

**详细分析**:

{summary}
"""
    return enhanced_summary

def generate_core_parameters_section(json_data, metadata):
    """生成核心参数部分（最重要的参数）"""
    detailed = json_data.get("detailed_data", {})
    content = []
    
    # 温度相关核心参数
    temp_data = detailed.get("温度特征", {})
    if temp_data:
        content.append("### 3.1 温度特征")
        content.append(f"- **2米平均气温**: {temp_data.get('t2m_mean', {}).get('raw', 'N/A')}")
        content.append(f"- **气温范围**: {temp_data.get('t2m_min', {}).get('raw', 'N/A')} ~ {temp_data.get('t2m_max', {}).get('raw', 'N/A')}")
        content.append(f"- **温度递减率**: {temp_data.get('temp_lapse_rate', {}).get('raw', 'N/A')}")
        content.append("")
    
    # 对流能量核心参数
    cape_data = detailed.get("对流能量", {})
    if cape_data:
        content.append("### 3.2 对流能量")
        content.append(f"- **最大对流有效位能(CAPE)**: {cape_data.get('cape_max', {}).get('raw', 'N/A')}")
        content.append(f"- **稳定度指数**: {cape_data.get('stability_index', {}).get('raw', 'N/A')}")
        content.append("")
    
    # 降水核心参数
    precip_data = detailed.get("降水特征", {})
    if precip_data:
        content.append("### 3.3 降水特征")
        content.append(f"- **3小时最大降水量**: {precip_data.get('tp_max', {}).get('raw', 'N/A')}")
        content.append(f"- **强降水比例**: {precip_data.get('heavy_rain_ratio', {}).get('raw', 'N/A')}")
        content.append("")
    
    # 涡度和垂直运动核心参数
    vorticity_data = detailed.get("涡度特征", {})
    vertical_data = detailed.get("垂直运动", {})
    if vorticity_data or vertical_data:
        content.append("### 3.4 动力条件")
        if vorticity_data:
            content.append(f"- **最大涡度**: {vorticity_data.get('vor_max', {}).get('raw', 'N/A')}")
            content.append(f"- **正涡度比例**: {vorticity_data.get('vor_positive_ratio', {}).get('raw', 'N/A')}")
        if vertical_data:
            content.append(f"- **400hPa最小垂直上升速度**: {vertical_data.get('w4_min', {}).get('raw', 'N/A')}")
        content.append("")
    
    # 如果没有核心参数
    if not content:
        return "**无核心参数数据**"
    
    return "\n".join(content)

def generate_detailed_section(json_data, metadata):
    """生成详细数据部分"""
    detailed = json_data.get("detailed_data", {})
    content = []
    
    categories = [
        ("温度特征", detailed.get("温度特征", {})),
        ("对流能量", detailed.get("对流能量", {})),
        ("湿度特征", detailed.get("湿度特征", {})),
        ("风场特征", detailed.get("风场特征", {})),
        ("垂直运动", detailed.get("垂直运动", {})),
        ("涡度特征", detailed.get("涡度特征", {})),
        ("水汽输送", detailed.get("水汽输送", {})),
        ("位势高度", detailed.get("位势高度", {})),
        ("降水特征", detailed.get("降水特征", {}))
    ]
    
    section_number = 4.1
    
    for category_name, category_data in categories:
        if category_data:
            content.append(f"### {section_number} {category_name}")
            section_number += 0.1
            
            for param, data in category_data.items():
                if isinstance(data, dict) and data.get("raw"):
                    param_name = translate_param_name(param, category_name)
                    # 清理单位显示
                    raw_value = data['raw']
                    if isinstance(raw_value, str):
                        raw_value = raw_value.replace(" (°C)", "°C").replace(" (J/kg)", "J/kg")
                        raw_value = raw_value.replace(" (m/s)", "m/s").replace(" (1/s)", "1/s")
                        raw_value = raw_value.replace(" (kg/kg)", "kg/kg").replace(" (Pa/s)", "Pa/s")
                        raw_value = raw_value.replace(" (kg/(m²·s", "kg/(m²·s)").replace(" (m²/s²)", "m²/s²")
                    
                    # 提取数值用于排序
                    if isinstance(raw_value, str):
                        num_match = re.search(r"[-+]?\d*\.\d+|\d+", raw_value)
                    else:
                        num_match = None
                    
                    if num_match:
                        try:
                            value = float(num_match.group())
                            # 高亮显示重要参数
                            if is_important_parameter(param, value):
                                content.append(f"- **{param_name}**: **{raw_value}**")
                            else:
                                content.append(f"- {param_name}: {raw_value}")
                        except:
                            content.append(f"- {param_name}: {raw_value}")
                    else:
                        content.append(f"- {param_name}: {raw_value}")
            content.append("")
    
    # 如果没有详细数据
    if not content:
        return "**无详细数据**"
    
    return "\n".join(content)

def generate_keywords_section(metadata, json_data):
    """生成关键词部分"""
    summary = json_data.get("summary", "")
    detailed = json_data.get("detailed_data", {})
    
    keywords = []
    
    # 时间关键词
    month = metadata.get('month', 0)
    if 1 <= month <= 12:
        month_names = ["一月", "二月", "三月", "四月", "五月", "六月", 
                      "七月", "八月", "九月", "十月", "十一月", "十二月"]
        keywords.append(month_names[month-1])
    
    # 季节关键词
    season = metadata.get('season', '')
    season_map = {
        "summer": ["夏季", "夏天"],
        "winter": ["冬季", "冬天"],
        "spring": ["春季", "春天"],
        "autumn": ["秋季", "秋天"]
    }
    if season in season_map:
        keywords.extend(season_map[season])
    
    # 时间点关键词
    hour = metadata.get('hour', 0)
    if 0 <= hour < 6:
        keywords.extend(["凌晨", "深夜"])
    elif 6 <= hour < 12:
        keywords.extend(["上午", "早晨"])
    elif 12 <= hour < 18:
        keywords.extend(["下午", "午后"])
    else:
        keywords.extend(["晚上", "夜间"])
    
    # 从摘要中提取天气现象关键词
    if isinstance(summary, str):
        weather_keywords = ["强降水", "对流", "涡旋", "上升运动", "水汽辐合", 
                           "红色预警", "高对流能量", "次生灾害", "大风", "暴雨"]
        for keyword in weather_keywords:
            if keyword in summary:
                keywords.append(keyword)
    
    # 从核心参数中提取技术关键词
    if detailed.get("温度特征", {}).get("temp_lapse_rate", {}).get("raw"):
        keywords.append("温度递减率")
    if detailed.get("对流能量", {}).get("cape_max", {}).get("raw"):
        keywords.append("对流有效位能")
        keywords.append("CAPE")
    if detailed.get("涡度特征", {}).get("vor_max", {}).get("raw"):
        keywords.append("涡度")
    
    # 去重
    keywords = list(set(keywords))
    
    if not keywords:
        return "**无可用关键词**"
    
    # 格式化关键词
    keywords_section = "以下关键词可用于检索本天气记录：\n\n"
    keywords_section += ", ".join([f"`{kw}`" for kw in keywords])
    
    return keywords_section

def format_datetime(datetime_str):
    """格式化日期时间"""
    if not datetime_str:
        return "未知时间"
    
    try:
        if "T" in datetime_str:
            # 处理可能的时区信息
            if datetime_str.endswith("Z"):
                dt_str = datetime_str.replace("Z", "+00:00")
            else:
                dt_str = datetime_str
            
            # 尝试解析ISO格式
            try:
                dt = datetime.fromisoformat(dt_str)
            except:
                # 如果失败，尝试移除微秒
                dt_str = re.sub(r'\.\d+', '', dt_str)
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        else:
            # 尝试其他格式
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
                try:
                    dt = datetime.strptime(datetime_str, fmt)
                    break
                except:
                    continue
            else:
                return datetime_str
    except Exception as e:
        print(f"时间格式化错误: {datetime_str}, 错误: {e}")
        return datetime_str
    
    return dt.strftime("%Y年%m月%d日%H时")

def translate_param_name(param_name, category):
    """翻译参数名称"""
    translations = {
        "t2m_mean": "2米平均气温",
        "t2m_max": "2米最高气温",
        "t2m_min": "2米最低气温",
        "t4_mean": "400hPa气温",
        "t7_mean": "700hPa气温",
        "t8_mean": "850hPa气温",
        "t9_mean": "900hPa气温",
        "temp_lapse_rate": "温度递减率",
        "cape_max": "最大对流有效位能(CAPE)",
        "cape_mean": "平均对流有效位能",
        "cape_high_ratio": "对流有效位能高于平均值的比例",
        "stability_index": "稳定度指数",
        "q7_mean": "700hPa平均比湿",
        "q7_max": "700hPa最大比湿",
        "q8_mean": "850hPa平均比湿",
        "q_low_mean": "低层平均比湿",
        "moisture_index": "湿度指数",
        "u2_mean": "2米平均U向风速",
        "u2_max": "2米最大U向风速",
        "v7_mean": "10米平均V向风速",
        "v7_max": "10米最大V向风速",
        "low_level_wind_speed": "低层风速",
        "w4_min": "400hPa最小垂直上升速度",
        "w4_mean": "400hPa平均垂直上升速度",
        "w4_upward_ratio": "400hPa上升运动区面积占比",
        "vv9_min": "900hPa最小垂直上升速度",
        "vv9_mean": "900hPa平均垂直上升速度",
        "vor_max": "最大涡度",
        "vor_min": "最小涡度",
        "vor_mean": "平均涡度",
        "vor_positive_ratio": "正涡度比例",
        "vidmf_min": "最小水汽通量散度",
        "vidmf_mean": "平均水汽通量散度",
        "vidmf_convergence_ratio": "水汽通量散度收敛率",
        "z2_mean": "200hPa平均位势高度",
        "z2_max": "200hPa最大位势高度",
        "z2_min": "200hPa最小位势高度",
        "z4_min": "400hPa最小位势高度",
        "z4_mean": "400hPa平均位势高度",
        "z4_max": "400hPa最大位势高度",
        "z5_min": "500hPa最小位势高度",
        "z5_mean": "500hPa平均位势高度",
        "z5_max": "500hPa最大位势高度",
        "tp_max": "3小时最大降水量",
        "tp_mean": "3小时平均降水量",
        "heavy_rain_ratio": "强降水比例",
        "moderate_rain_ratio": "中降水比例"
    }
    return translations.get(param_name, param_name)

def is_important_parameter(param_name, value):
    """判断是否为重要参数（需要高亮显示）"""
    important_params = {
        "cape_max": (1000, float('inf')),  # CAPE > 1000 J/kg
        "tp_max": (10, float('inf')),      # 降水 > 10mm
        "vor_max": (0.0001, float('inf')),  # 涡度 > 0.0001
        "w4_min": (-1.0, -0.5),           # 强烈上升运动
        "low_level_wind_speed": (8, float('inf'))  # 大风
    }
    
    if param_name in important_params:
        min_val, max_val = important_params[param_name]
        return min_val <= value <= max_val
    return False

# 主程序
if __name__ == "__main__":
    # 读取JSON文件
    input_file = "/home/guojn/code/qikeyuan/DATA/processed_meteorological_data/meteorological_data.json"  # 替换为您的实际文件路径
    output_directory = "/home/guojn/code/qikeyuan/DATA/processed_meteorological_data/3_data"  # 输出目录
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"加载数据完成，数据类型: {type(data)}")
        
        if isinstance(data, list):
            print(f"数据是列表，包含 {len(data)} 个元素")
            
            # 批量处理列表
        output_files = process_json_list_to_markdown(data, output_directory)
            
        print(f"\n处理完成！共生成 {len(output_files)} 个Markdown文件")
        print(f"文件保存在: {os.path.abspath(output_directory)}")
            
            # 显示前几个文件名
        if output_files:
            print("\n前5个生成的文件:")
            for file_path in output_files[:5]:
                print(f"  - {os.path.basename(file_path)}")
            if len(output_files) > 5:
                print(f"  ... 以及另外 {len(output_files)-5} 个文件")
            
    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 不存在")
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")
    except Exception as e:
        print(f"错误：{type(e).__name__} - {e}")
#顺利的话，应该会生成一个文件夹，装载.md文件，即该目录下的/markdown
#该文件夹下会装满所有可用时间段的气象特征信息，由于我的数据覆盖有两万多条，就只放一个同名的空markdown文件夹示意