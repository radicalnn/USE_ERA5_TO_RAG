import random
import json
import itertools
from typing import List, Dict, Tuple, Set
from datetime import datetime
import numpy as np

# 中文同义词词典
SYNONYMS = {
    "今天": ["今日", "当天", "本日", "当前", "此刻"],
    "气温": ["温度", "热度", "气候温度", "气温值", "温度值"],
    "湿度": ["湿润度", "潮湿程度", "水分含量", "湿气", "湿度值"],
    "风速": ["风力", "风势", "风强", "风速值", "风的大小"],
    "降水": ["降雨", "下雨", "降水", "降雨量", "下雨量"],
    "对流": ["对流活动", "对流过程", "对流现象", "对流体"],
    "涡度": ["旋转度", "旋转强度", "涡旋强度", "涡旋度"],
    "多少": ["几多", "若干", "多大", "多少数值", "数值是多少"],
    "怎么样": ["如何", "怎样", "情况如何", "状况如何", "什么情况"],
    "平均": ["均值", "平均值", "平均数", "平均数值"],
    "最大": ["最高", "最大值", "最大数值", "顶峰", "峰值"],
    "最小": ["最低", "最小值", "最小数值", "谷值", "底部"],
    "高": ["大", "强", "高值", "较高"],
    "低": ["小", "弱", "低值", "较低"],
    "垂直": ["竖直", "垂直方向", "铅直", "垂直向"],
    "上升": ["抬升", "向上", "升起", "上涌"],
    "下降": ["下沉", "向下", "降落", "下落"],
    "比例": ["比率", "占比", "百分比", "比例值"],
    "指数": ["指标", "系数", "参数", "数值"],
    "有效": ["有用", "有效果", "有效应的", "有效的"],
    "位能": ["势能", "能量", "潜能", "潜在能量"],
    "对流有效位能": ["CAPE", "对流潜能", "对流不稳定能量"],
    "水汽": ["水汽含量", "水汽量", "水汽浓度", "湿度"],
    "通量": ["流量", "通量值", "传输量", "输送量"],
    "散度": ["发散度", "散开程度", "扩散度"],
    "位势高度": ["位势", "位势场", "位势值", "高度场"],
    "降水量": ["降雨量", "降水总量", "降水深度", "雨量"],
    "强降水": ["暴雨", "大雨", "强降雨", "强雨"],
    "中降水": ["中雨", "中等降水", "中等降雨", "中强度降水"],
    "稳定度": ["稳定性", "稳定程度", "稳定指数", "稳定系数"],
    "比湿": ["比湿度", "湿度比", "水汽比", "混合比"],
    "垂直上升速度": ["垂直速度", "上升速度", "抬升速度", "垂向速度"],
    "涡度": ["涡旋度", "旋转量", "旋度", "涡量"],
    "水汽通量散度": ["水汽输送散度", "水汽辐散", "水汽通量扩散"],
    "位势高度": ["位势场", "重力位势", "位势面", "等位势面"],
}

# 查询模板
QUERY_TEMPLATES = [
    "{}是多少？",
    "请问{}是多少？",
    "{}的数值是多少？",
    "{}多少？",
    "{}怎么样？",
    "{}的情况如何？",
    "{}的数值？",
    "{}的数值是多少呢？",
    "{}的数值是多少啊？",
    "{}的数值是？",
    "{}的数值多少？",
    "{}怎么样？数值是多少？",
    "{}的数值如何？",
    "{}是多少数值？",
    "{}的具体数值？",
    "{}的数值具体是多少？",
    "{}的数值是多少？请告诉我",
    "{}的数值是多少？谢谢",
    "{}的数值是多少？请问",
    "{}是多少？能告诉我吗？",
    "{}是多少？请帮忙查询",
    "{}是多少？我想知道",
    "{}是多少？麻烦告诉我",
    "{}的数值是多少？请提供",
    "{}的数值是多少？谢谢提供",
    "{}是多少？请告知",
    "{}是多少？请告知数值",
    "{}是多少？我需要知道",
    "{}是多少？想了解一下",
    "{}的数值是多少？了解下",
    "{}的数值是多少？想知道",
    "{}是多少？可以告诉我吗？",
    "{}是多少？能提供吗？",
    "{}是多少？能查询到吗？",
    "{}的数值是多少？查询一下",
    "{}的数值是多少？查一下",
    "{}的数值是多少？帮忙查一下",
    "{}是多少？帮忙查询下",
    "{}是多少？查询结果如何？",
    "{}是多少？有结果吗？",
]

# 组合查询模板
COMBINATION_TEMPLATES = [
    "{}和{}怎么样？",
    "{}、{}分别是多少？",
    "{}和{}的情况如何？",
    "{}、{}的数值是多少？",
    "{}和{}的数值？",
    "{}、{}怎么样？数值是多少？",
    "{}和{}的数值是多少？请告诉我",
    "{}、{}的数值是多少？谢谢",
    "{}和{}的数值是多少？请问",
    "{}、{}的数值是多少？能告诉我吗？",
    "{}和{}的数值是多少？请帮忙查询",
    "{}、{}的数值是多少？我想知道",
    "{}和{}的数值是多少？麻烦告诉我",
    "{}、{}的数值是多少？请提供",
    "{}和{}的数值是多少？谢谢提供",
    "{}、{}的数值是多少？请告知",
    "{}和{}的数值是多少？请告知数值",
    "{}、{}的数值是多少？我需要知道",
    "{}和{}的数值是多少？想了解一下",
    "{}、{}的数值是多少？了解下",
    "{}和{}的数值是多少？想知道",
    "{}、{}的数值是多少？可以告诉我吗？",
    "{}和{}的数值是多少？能提供吗？",
    "{}、{}的数值是多少？能查询到吗？",
    "{}和{}的数值是多少？查询一下",
    "{}、{}的数值是多少？查一下",
    "{}和{}的数值是多少？帮忙查一下",
    "{}、{}的数值是多少？帮忙查询下",
    "{}和{}的数值是多少？查询结果如何？",
    "{}、{}的数值是多少？有结果吗？",
]

def synonym_replacement(text: str, probability: float = 0.3) -> str:
    """同义词替换增强"""
    words = list(text)
    new_words = []
    i = 0
    while i < len(words):
        # 尝试匹配1-4个字符的同义词
        matched = False
        for length in range(4, 0, -1):
            if i + length <= len(words):
                word = ''.join(words[i:i+length])
                if word in SYNONYMS and random.random() < probability:
                    new_words.append(random.choice(SYNONYMS[word]))
                    i += length
                    matched = True
                    break
        if not matched:
            new_words.append(words[i])
            i += 1
    return ''.join(new_words)

def generate_variations(param_name: str, num_variations: int = 20) -> List[Tuple[str, str]]:
    """为单个参数生成多个查询变体"""
    variations = []
    
    # 基础模板生成
    for template in QUERY_TEMPLATES[:min(15, len(QUERY_TEMPLATES))]:
        query = template.format(param_name)
        variations.append((query, param_name))
    
    # 特殊参数处理
    if "CAPE" in param_name or "对流有效位能" in param_name:
        cape_variations = [
            (f"{param_name}值是多少？", f"{param_name} CAPE J/kg"),
            (f"{param_name}是多少？", f"{param_name} CAPE"),
            ("CAPE值是多少？", "CAPE 对流有效位能"),
            ("对流能量多少？", "CAPE 对流能量"),
        ]
        variations.extend(cape_variations)
    
    if "气温" in param_name:
        temp_variations = [
            (f"{param_name}多少度？", f"{param_name} 摄氏度"),
            (f"{param_name}的温度？", f"{param_name} 温度"),
            ("温度是多少？", f"{param_name} 气温"),
        ]
        variations.extend(temp_variations)
    
    if "风速" in param_name:
        wind_variations = [
            (f"{param_name}多大？", f"{param_name} m/s"),
            (f"{param_name}的风力？", f"{param_name} 风速"),
            ("风力多少？", f"{param_name} 风"),
        ]
        variations.extend(wind_variations)
    
    if "降水" in param_name:
        rain_variations = [
            (f"{param_name}多少毫米？", f"{param_name} mm"),
            (f"{param_name}的雨量？", f"{param_name} 降水量"),
            ("雨量多少？", f"{param_name} 降雨"),
        ]
        variations.extend(rain_variations)
    
    # 同义词替换增强
    enhanced_variations = []
    for original, rewritten in variations:
        # 保留原始版本
        enhanced_variations.append((original, rewritten))
        
        # 同义词替换版本
        for _ in range(2):
            new_original = synonym_replacement(original, probability=0.2)
            if new_original != original:
                enhanced_variations.append((new_original, rewritten))
    
    # 添加语气词变体
    mood_words = ["请问", "请告诉我", "请问一下", "帮我查询", "我想知道", "请帮忙查询", "麻烦查询", "查询一下"]
    final_variations = []
    for original, rewritten in enhanced_variations:
        final_variations.append((original, rewritten))
        
        # 添加语气词
        for mood in random.sample(mood_words, 2):
            mood_query = f"{mood}{original}"
            final_variations.append((mood_query, rewritten))
    
    # 去重并限制数量
    seen = set()
    unique_variations = []
    for original, rewritten in final_variations:
        if original not in seen and len(unique_variations) < num_variations:
            seen.add(original)
            unique_variations.append((original, rewritten))
    
    return unique_variations

def generate_param_to_queries() -> Dict[str, List[Tuple[str, str]]]:
    """生成参数到查询的映射"""
    # 定义所有参数
    all_params = {
        "2米平均气温": "2米平均气温 温度 摄氏度",
        "2米最高气温": "2米最高气温 温度 摄氏度",
        "2米最低气温": "2米最低气温 温度 摄氏度",
        "400hPa气温": "400hPa气温 高空温度",
        "700hPa气温": "700hPa气温 中空温度",
        "850hPa气温": "850hPa气温 低空温度",
        "900hPa气温": "900hPa气温 低空温度",
        "850hpa与500hpa的平均温差": "温度递减率 温差 垂直梯度",
        "最大对流有效位能": "最大对流有效位能 CAPE J/kg 对流能量",
        "平均对流有效位能": "平均对流有效位能 CAPE 对流能量",
        "对流有效位能高于平均值的比例": "对流有效位能高于平均值的比例 高CAPE比例",
        "稳定度指标": "稳定度指数 稳定度 大气稳定度",
        "700hPa平均比湿": "700hPa平均比湿 中空湿度 比湿",
        "700hPa最大比湿": "700hPa最大比湿 中空湿度 比湿",
        "850hPa平均比湿": "850hPa平均比湿 低空湿度 比湿",
        "低层平均比湿": "低层平均比湿 近地面湿度 比湿",
        "湿度指数": "湿度指数 湿度指标 湿润度",
        "2米平均U向风速": "2米平均U向风速 U向风 东西风向 m/s",
        "2米最大U向风速": "2米最大U向风速 U向风 东西风向",
        "10米平均V向风速": "10米平均V向风速 V向风 南北风向 m/s",
        "10米最大V向风速": "10米最大V向风速 V向风 南北风向",
        "低层风速": "低层风速 近地面风 m/s 风场",
        "400hPa最小垂直上升速度": "400hPa最小垂直上升速度 垂直速度 Pa/s 上升运动",
        "400hPa平均垂直上升速度": "400hPa平均垂直上升速度 垂直速度 上升运动",
        "400hPa上升运动区面积占比": "400hPa上升运动区面积占比 上升运动 面积比例",
        "900hPa最小垂直上升速度": "900hPa最小垂直上升速度 垂直速度 上升运动",
        "900hPa平均垂直上升速度": "900hPa平均垂直上升速度 垂直速度 上升运动",
        "最大涡度": "最大涡度 涡度 旋转强度 1/s",
        "最小涡度": "最小涡度 涡度 旋转强度",
        "平均涡度": "平均涡度 涡度 旋转强度",
        "正涡度比例": "正涡度比例 正涡度 涡度比例",
        "最小水汽通量散度": "最小水汽通量散度 水汽通量 散度 水汽",
        "平均水汽通量散度": "平均水汽通量散度 水汽通量 散度 水汽",
        "水汽通量散度收敛率": "水汽通量散度收敛率 水汽通量 收敛 水汽",
        "200hPa平均位势高度": "200hPa平均位势高度 位势高度 高空位势",
        "200hPa最大位势高度": "200hPa最大位势高度 位势高度 高空位势",
        "200hPa最小位势高度": "200hPa最小位势高度 位势高度 高空位势",
        "400hPa平均位势高度": "400hPa平均位势高度 位势高度 高空位势",
        "400hPa最大位势高度": "400hPa最大位势高度 位势高度 高空位势",
        "400hPa最小位势高度": "400hPa最小位势高度 位势高度 高空位势",
        "500hPa平均位势高度": "500hPa平均位势高度 位势高度 中空位势",
        "500hPa最大位势高度": "500hPa最大位势高度 位势高度 中空位势",
        "500hPa最小位势高度": "500hPa最小位势高度 位势高度 中空位势",
        "3小时最大降水量": "3小时最大降水量 降水 mm 降雨",
        "3小时平均降水量": "3小时平均降水量 降水 降雨 平均雨量",
        "强降水比例": "强降水比例 强降水 暴雨比例 大雨比例",
        "中降水比例": "中降水比例 中降水 中雨比例 中等降水",
    }
    
    param_to_queries = {}
    for param, rewritten_base in all_params.items():
        variations = generate_variations(param, num_variations=25)
        param_to_queries[param] = [(original, rewritten_base) for original, _ in variations]
    
    return param_to_queries

def generate_combination_queries(param_names: List[str], num_combinations: int = 1000) -> List[Tuple[str, str]]:
    """生成组合查询"""
    combinations = []
    seen_combinations = set()
    
    # 两参数组合
    for _ in range(num_combinations // 2):
        if len(param_names) < 2:
            break
            
        param1, param2 = random.sample(param_names, 2)
        combination_key = frozenset([param1, param2])
        
        if combination_key in seen_combinations:
            continue
            
        # 从模板中选择
        template = random.choice(COMBINATION_TEMPLATES)
        query = template.format(param1, param2)
        rewritten = f"{param1} {param2}"
        
        combinations.append((query, rewritten))
        seen_combinations.add(combination_key)
        
        # 同义词替换版本
        for _ in range(2):
            new_query = synonym_replacement(query, probability=0.15)
            if new_query != query:
                combinations.append((new_query, rewritten))
    
    # 三参数组合
    for _ in range(num_combinations // 4):
        if len(param_names) < 3:
            break
            
        selected = random.sample(param_names, 3)
        combination_key = frozenset(selected)
        
        if combination_key in seen_combinations:
            continue
        
        # 生成查询
        query = f"{selected[0]}、{selected[1]}和{selected[2]}怎么样？"
        rewritten = " ".join(selected)
        
        combinations.append((query, rewritten))
        seen_combinations.add(combination_key)
    
    # 四参数组合
    for _ in range(num_combinations // 8):
        if len(param_names) < 4:
            break
            
        selected = random.sample(param_names, 4)
        combination_key = frozenset(selected)
        
        if combination_key in seen_combinations:
            continue
        
        # 生成查询
        query = f"{selected[0]}、{selected[1]}、{selected[2]}和{selected[3]}的数值是多少？"
        rewritten = " ".join(selected)
        
        combinations.append((query, rewritten))
        seen_combinations.add(combination_key)
    
    return combinations

def generate_training_data_from_metadata(
    num_samples_per_param: int = 30, 
    num_combination_queries: int = 1500
) -> List[Dict]:
    """从现有数据自动生成训练样本（增强版）"""
    
    # 生成参数到查询的映射
    param_to_queries = generate_param_to_queries()
    param_names = list(param_to_queries.keys())
    
    print(f"总参数数量: {len(param_names)}")
    print(f"每个参数样本数: {num_samples_per_param}")
    
    # 生成训练样本
    pairs = []
    
    # 1. 单参数查询
    for param, query_variants in param_to_queries.items():
        # 为每个参数选择指定数量的查询变体
        if len(query_variants) <= num_samples_per_param:
            selected_queries = query_variants
        else:
            selected_queries = random.sample(query_variants, num_samples_per_param)
        
        for original_query, rewritten_query in selected_queries:
            pairs.append({
                "original": original_query,
                "rewritten": rewritten_query,
                "param": param
            })
    
    print(f"单参数查询生成: {len(pairs)} 个样本")
    
    # 2. 组合查询
    combination_queries = generate_combination_queries(param_names, num_combination_queries)
    
    for original, rewritten in combination_queries:
        pairs.append({
            "original": original,
            "rewritten": rewritten,
            "param": "combined"
        })
    
    print(f"组合查询生成: {len(combination_queries)} 个样本")
    
    # 3. 预定义的组合查询（保持多样性）
    predefined_combinations = [
        ("今天的气温和湿度怎么样？", "2米平均气温 2米最高气温 2米最低气温 湿度指数"),
        ("温度和降水是多少？", "2米平均气温 3小时最大降水量"),
        ("温度和风场怎么样？", "2米平均气温 低层风速"),
        ("温度和垂直运动如何？", "2米平均气温 400hPa平均垂直上升速度"),
        ("温度和对流能量多少？", "2米平均气温 最大对流有效位能"),
        ("湿度和降水怎么样？", "低层平均比湿 3小时最大降水量"),
        ("湿度和垂直运动如何？", "低层平均比湿 400hPa平均垂直上升速度"),
        ("湿度和风场怎么样？", "低层平均比湿 低层风速"),
        ("风和对流能量多少？", "低层风速 最大对流有效位能"),
        ("风场和垂直运动怎么样？", "低层风速 400hPa平均垂直上升速度"),
        ("风场和降水怎么样？", "低层风速 3小时最大降水量"),
        ("对流和降水情况如何？", "最大对流有效位能 3小时最大降水量 强降水比例"),
        ("对流和垂直运动？", "最大对流有效位能 400hPa平均垂直上升速度"),
        ("对流和稳定度？", "最大对流有效位能 稳定度指标"),
        ("垂直运动和降水？", "400hPa平均垂直上升速度 3小时最大降水量"),
        ("垂直运动和湿度？", "400hPa平均垂直上升速度 低层平均比湿"),
        ("涡度和降水？", "最大涡度 3小时最大降水量"),
        ("涡度和垂直运动？", "最大涡度 400hPa平均垂直上升速度"),
        ("涡度和风场？", "最大涡度 低层风速"),
        ("位势高度和温度？", "500hPa平均位势高度 2米平均气温"),
        ("位势高度和风场？", "500hPa平均位势高度 低层风速"),
        ("位势高度和湿度？", "500hPa平均位势高度 低层平均比湿"),
        ("今天的天气综合情况？", "2米平均气温 低层风速 3小时最大降水量 最大对流有效位能"),
        ("对流天气的综合指标？", "最大对流有效位能 400hPa平均垂直上升速度 最大涡度 低层平均比湿"),
        ("暴雨天气条件？", "3小时最大降水量 低层平均比湿 400hPa平均垂直上升速度 低层风速"),
        ("高温高湿天气？", "2米平均气温 低层平均比湿 2米最高气温 湿度指数"),
        ("大风降温天气？", "低层风速 2米平均气温 温度递减率"),
        ("雷暴天气潜力？", "最大对流有效位能 低层平均比湿 400hPa平均垂直上升速度 稳定度指标"),
    ]
    
    for original, rewritten in predefined_combinations:
        pairs.append({
            "original": original,
            "rewritten": rewritten,
            "param": "predefined_combined"
        })
    
    print(f"预定义组合查询生成: {len(predefined_combinations)} 个样本")
    
    # 4. 数据增强：对已有样本进行同义词替换
    original_count = len(pairs)
    augmented_pairs = []
    
    for pair in pairs:
        augmented_pairs.append(pair)
        
        # 对部分样本进行增强
        if random.random() < 0.3:  # 30%的样本进行增强
            new_original = synonym_replacement(pair["original"], probability=0.25)
            if new_original != pair["original"]:
                augmented_pairs.append({
                    "original": new_original,
                    "rewritten": pair["rewritten"],
                    "param": f"{pair['param']}_aug"
                })
    
    print(f"数据增强后: {len(augmented_pairs)} 个样本 (增强前: {original_count} 个样本)")
    
    # 打乱数据
    random.shuffle(augmented_pairs)
    
    return augmented_pairs

def save_training_data(pairs: List[Dict], output_path: str = "weather_query_rewrite_data_large.json"):
    """保存训练数据到JSON文件"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    print(f"\n 训练数据生成完成！")
    print(f"   保存到: {output_path}")
    print(f"   总样本数: {len(pairs):,} 个")
    
    # 统计数据
    param_counts = {}
    for pair in pairs:
        param = pair["param"]
        if param.endswith("_aug"):
            param = param[:-4]  # 去掉_aug后缀
        param_counts[param] = param_counts.get(param, 0) + 1
    
    print(f"\n参数分布统计 (前20个):")
    sorted_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for param, count in sorted_params:
        print(f"  {param}: {count:4d} 个样本 ({count/len(pairs)*100:.1f}%)")
    
    # 计算平均查询长度
    orig_lengths = [len(pair["original"]) for pair in pairs]
    rewrite_lengths = [len(pair["rewritten"]) for pair in pairs]
    
    print(f"\n查询长度统计:")
    print(f"  原始查询平均长度: {np.mean(orig_lengths):.1f} 字符")
    print(f"  重写查询平均长度: {np.mean(rewrite_lengths):.1f} 字符")
    print(f"  原始查询最短长度: {np.min(orig_lengths)} 字符")
    print(f"  原始查询最长长度: {np.max(orig_lengths)} 字符")
    
    return pairs

# 生成和保存训练数据
if __name__ == "__main__":
    print("=" * 60)
    print("气象查询重写训练数据生成器 (增强版)")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(42)
    
    # 生成大规模训练数据
    print("\n开始生成训练数据...")
    training_pairs = generate_training_data_from_metadata(
        num_samples_per_param=30,      # 每个参数30个样本
        num_combination_queries=1500   # 1500个组合查询
    )
    
    # 保存数据
    save_training_data(training_pairs, "weather_query_rewrite_data_large.json")
    
    # 输出前5个样本作为示例
    print("\n" + "=" * 60)
    print("前5个训练样本示例:")
    print("=" * 60)
    
    for i, pair in enumerate(training_pairs[:5]):
        print(f"\n样本 {i+1}:")
        print(f"  参数: {pair['param']}")
        print(f"  原始查询: {pair['original']}")
        print(f"  重写查询: {pair['rewritten']}")
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)