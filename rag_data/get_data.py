# %%
import xarray as xr
import numpy as np
import pandas as pd
import random
from datetime import datetime
from pathlib import Path

def extract_features_single_timestep(data_slice):
    """从单个时次的数据切片中提取特征"""
    features = {}
    
    try:
        # 1. 提取时间信息
        time_val = data_slice['time'].values
        features['time'] = str(time_val)
        
        # 2. 提取基本统计特征
        if 'cape' in data_slice:
            cape_data = data_slice['cape']
            features['cape_max'] = float(cape_data.max().values)
            features['cape_mean'] = float(cape_data.mean().values)
            cape_high_mask = cape_data > 1000
            features['cape_high_ratio'] = float(cape_high_mask.sum() / cape_data.size)
        
        if 'q7' in data_slice:
            q7_data = data_slice['q7']
            features['q7_mean'] = float(q7_data.mean().values)
            features['q7_max'] = float(q7_data.max().values)
        
        if 'q8' in data_slice:
            q8_data = data_slice['q8']
            features['q8_mean'] = float(q8_data.mean().values)
            if 'q7' in data_slice:
                features['q_low_mean'] = float(0.5 * (q7_data + q8_data).mean().values)
        
        if 't2m' in data_slice:
            t2m_data = data_slice['t2m'] - 273.15
            features['t2m_mean'] = float(t2m_data.mean().values)
            features['t2m_max'] = float(t2m_data.max().values)
            features['t2m_min'] = float(t2m_data.min().values)
        
        for t_var in ['t4', 't7', 't8', 't9']:
            if t_var in data_slice:
                t_data = data_slice[t_var] - 273.15
                features[f'{t_var}_mean'] = float(t_data.mean().values)
        
        if 't8' in data_slice and 't4' in data_slice:
            t8_data = data_slice['t8'] - 273.15
            t4_data = data_slice['t4'] - 273.15
            features['temp_lapse_rate'] = float((t8_data - t4_data).mean().values)
        
        if 'u2' in data_slice:
            u2_data = data_slice['u2']
            features['u2_mean'] = float(u2_data.mean().values)
            features['u2_max'] = float(u2_data.max().values)
        
        if 'v7' in data_slice:
            v7_data = data_slice['v7']
            features['v7_mean'] = float(v7_data.mean().values)
            features['v7_max'] = float(v7_data.max().values)
        
        if 'u2' in data_slice and 'v7' in data_slice:
            u2_mean = features.get('u2_mean', 0)
            v7_mean = features.get('v7_mean', 0)
            features['low_level_wind_speed'] = float(np.sqrt(u2_mean**2 + v7_mean**2))
        
        if 'vidmf' in data_slice:
            vidmf_data = data_slice['vidmf']
            features['vidmf_min'] = float(vidmf_data.min().values)
            features['vidmf_mean'] = float(vidmf_data.mean().values)
            vidmf_convergence_mask = vidmf_data < -1e-6
            features['vidmf_convergence_ratio'] = float(vidmf_convergence_mask.sum() / vidmf_data.size)
        
        if 'vor' in data_slice:
            vor_data = data_slice['vor']
            features['vor_max'] = float(vor_data.max().values)
            features['vor_min'] = float(vor_data.min().values)
            features['vor_mean'] = float(vor_data.mean().values)
            vor_positive_mask = vor_data > 1e-5
            features['vor_positive_ratio'] = float(vor_positive_mask.sum() / vor_data.size)
        
        if 'vv4' in data_slice:
            w4_data = data_slice['vv4']
            features['w4_min'] = float(w4_data.min().values)
            features['w4_mean'] = float(w4_data.mean().values)
            w4_upward_mask = w4_data < -0.1
            features['w4_upward_ratio'] = float(w4_upward_mask.sum() / w4_data.size)
        
        if 'vv9' in data_slice:
            vv9_data = data_slice['vv9']
            features['vv9_min'] = float(vv9_data.min().values)
            features['vv9_mean'] = float(vv9_data.mean().values)
        
        for z_var in ['z2', 'z4', 'z5']:
            if z_var in data_slice:
                z_data = data_slice[z_var]
                features[f'{z_var}_mean'] = float(z_data.mean().values)
                features[f'{z_var}_max'] = float(z_data.max().values)
                features[f'{z_var}_min'] = float(z_data.min().values)
        
        if 'z5' in data_slice:
            z5_data = data_slice['z5']
            if 'latitude' in z5_data.coords:
                lat = z5_data.latitude.values
                if len(lat) > 1:
                    z5_north = z5_data.sel(latitude=lat.max()).mean().values
                    z5_south = z5_data.sel(latitude=lat.min()).mean().values
                    lat_diff = lat.max() - lat.min()
                    if lat_diff > 0:
                        features['z5_gradient_ns'] = float((z5_north - z5_south) / lat_diff)
        
        if 'tp' in data_slice:
            tp_data = data_slice['tp']
            features['tp_max'] = float(tp_data.max().values)
            features['tp_mean'] = float(tp_data.mean().values)
            heavy_rain_mask = tp_data > 5
            features['heavy_rain_ratio'] = float(heavy_rain_mask.sum() / tp_data.size)
            moderate_rain_mask = tp_data > 10
            features['moderate_rain_ratio'] = float(moderate_rain_mask.sum() / tp_data.size)
        
        if 'cape_max' in features and 'temp_lapse_rate' in features:
            features['stability_index'] = features['cape_max'] / 1000 - features['temp_lapse_rate'] / 10
        
        if 'q_low_mean' in features and 'vidmf_convergence_ratio' in features:
            features['moisture_index'] = features['q_low_mean'] * 1000 + features['vidmf_convergence_ratio'] * 10
        
        if 'latitude' in data_slice.coords:
            features['lat_min'] = float(data_slice['latitude'].min().values)
            features['lat_max'] = float(data_slice['latitude'].max().values)
            features['lat_mean'] = float(data_slice['latitude'].mean().values)
        
        if 'longitude' in data_slice.coords:
            features['lon_min'] = float(data_slice['longitude'].min().values)
            features['lon_max'] = float(data_slice['longitude'].max().values)
            features['lon_mean'] = float(data_slice['longitude'].mean().values)
        
        # 添加时间特征
        try:
            time_str = features['time']
            dt = np.datetime64(time_str)
            month = dt.astype('datetime64[M]').astype(int) % 12 + 1
            hour = (dt - dt.astype('datetime64[D]')).astype('timedelta64[h]').astype(int)
            
            features['month'] = int(month)
            features['hour'] = int(hour)
            features['season'] = 'summer' if month in [6, 7, 8] else 'other'
            
        except Exception as e:
            features['month'] = 0
            features['hour'] = 0
            features['season'] = 'unknown'
        
        if features.get('month', 0) in [6, 7, 8] and 12 <= features.get('hour', 0) <= 18:
            features['is_summer_afternoon'] = 1
        else:
            features['is_summer_afternoon'] = 0
            
    except Exception as e:
        print(f"提取特征时出错: {e}")
        return None
    
    return features

def _generate_professional_style(input_text, features, year, month_cn, day, time_desc, hour_num,
                                t2m_mean, t2m_max, t2m_min, cape_max, tp_max, w4_min, 
                                q_low_mean, vor_max, is_heavy_rain, is_stormy, is_hot, is_cold):
    """专业风格：基于全部特征随机生成详细、准确、专业的天气分析"""
    output_parts = []
    
    # 从features中提取更多特征
    stability_index = features.get('stability_index', 0)
    temp_lapse_rate = features.get('temp_lapse_rate', 0)
    cape_high_ratio = features.get('cape_high_ratio', 0)
    moisture_index = features.get('moisture_index', 0)
    vidmf_convergence_ratio = features.get('vidmf_convergence_ratio', 0)
    vor_positive_ratio = features.get('vor_positive_ratio', 0)
    w4_upward_ratio = features.get('w4_upward_ratio', 0)
    heavy_rain_ratio = features.get('heavy_rain_ratio', 0)
    moderate_rain_ratio = features.get('moderate_rain_ratio', 0)
    is_summer_afternoon = features.get('is_summer_afternoon', 0)
    low_level_wind_speed = features.get('low_level_wind_speed', 0)
    z5_gradient_ns = features.get('z5_gradient_ns', 0)
    
    # 随机选择专业开头
    professional_openings = [
        f"{year}年{month_cn}{day}号{time_desc}{hour_num}时天气分析报告：",
        f"【天气分析】{month_cn}{day}号{time_desc}{hour_num}时：",
        f"气象数据分析（{month_cn}{day}号{time_desc}{hour_num}时）：",
        f"专业天气解读 - {month_cn}{day}号{time_desc}{hour_num}时：",
        f"综合气象分析报告（{year}年{month_cn}{day}号{time_desc}{hour_num}时）："
    ]
    output_parts.append(random.choice(professional_openings))
    
    # 温度场分析
    temp_analyses = []
    temp_range = f"{t2m_min:.1f}~{t2m_max:.1f}"
    
    if is_hot:
        temp_analyses = [
            f"高温天气特征明显，平均气温{t2m_mean:.1f}℃，极端温度{temp_range}℃。",
            f"受高温影响，近地面气温达{t2m_mean:.1f}℃，温度范围{temp_range}℃。",
            f"高温天气，平均气温{t2m_mean:.1f}℃，需注意防暑降温。"
        ]
    elif is_cold:
        temp_analyses = [
            f"低温天气，平均气温{t2m_mean:.1f}℃，温度范围{temp_range}℃。",
            f"气温偏低，平均{t2m_mean:.1f}℃，需注意保暖。",
            f"冷空气影响，气温{t2m_mean:.1f}℃，昼夜温差{abs(t2m_max-t2m_min):.1f}℃。"
        ]
    else:
        temp_analyses = [
            f"近地面气温{t2m_mean:.1f}℃（范围{temp_range}℃）。",
            f"温度适中，平均气温{t2m_mean:.1f}℃。",
            f"气温条件{t2m_mean:.1f}℃，温度分布{temp_range}℃。"
        ]
    output_parts.append(random.choice(temp_analyses))
    
    # 稳定度分析
    stability_analyses = []
    
    if cape_max > 2000:
        cape_desc = random.choice([
            f"大气极不稳定，CAPE值{cape_max:.0f} J/kg",
            f"强对流潜势极高，CAPE达{cape_max:.0f} J/kg",
            f"对流能量充沛，CAPE值{cape_max:.0f} J/kg"
        ])
    elif cape_max > 1000:
        cape_desc = random.choice([
            f"大气不稳定，CAPE值{cape_max:.0f} J/kg",
            f"有利于对流发展，CAPE值{cape_max:.0f} J/kg",
            f"对流潜势明显，CAPE值{cape_max:.0f} J/kg"
        ])
    elif cape_max > 500:
        cape_desc = f"具有一定对流潜势，CAPE值{cape_max:.0f} J/kg"
    else:
        cape_desc = f"大气稳定，CAPE值较低（{cape_max:.0f} J/kg）"
    
    if cape_high_ratio > 0.3:
        cape_ratio_desc = random.choice([
            f"，高CAPE区域占比{cape_high_ratio*100:.1f}%",
            f"，{cape_high_ratio*100:.1f}%区域对流潜势高",
            f"，广泛存在不稳定区域（{cape_high_ratio*100:.1f}%）"
        ])
        cape_desc += cape_ratio_desc
    
    if abs(stability_index) > 1.0:
        stability_desc = random.choice([
            f"，稳定度指数{stability_index:.2f}",
            f"，综合稳定度指标{stability_index:.2f}",
            f"，层结稳定度{stability_index:.2f}"
        ])
        cape_desc += stability_desc
    
    output_parts.append(cape_desc + "。")
    
    # 水汽条件分析
    moisture_parts = []
    
    if q_low_mean > 0.008:
        q_desc = random.choice([
            f"低层比湿{q_low_mean*1000:.1f} g/kg",
            f"水汽含量{q_low_mean*1000:.1f} g/kg",
            f"比湿条件{q_low_mean*1000:.1f} g/kg"
        ])
        moisture_parts.append(q_desc)
    
    if vidmf_convergence_ratio > 0.3:
        convergence_desc = random.choice([
            "水汽辐合明显",
            "存在水汽汇聚",
            "水汽辐合区显著"
        ])
        moisture_parts.append(convergence_desc + f"（{vidmf_convergence_ratio*100:.1f}%）")
    elif vidmf_convergence_ratio > 0.1:
        moisture_parts.append(f"部分水汽辐合（{vidmf_convergence_ratio*100:.1f}%）")
    
    if moisture_index > 5.0:
        moisture_parts.append("水汽条件充沛")
    elif moisture_index < 2.0:
        moisture_parts.append("水汽条件一般")
    
    if moisture_parts:
        moisture_intros = ["水汽条件：", "湿度特征：", "水汽分析："]
        output_parts.append(random.choice(moisture_intros) + "，".join(moisture_parts) + "。")
    
    # 动力条件分析
    dynamic_parts = []
    
    if w4_min < -0.5:
        w_desc = random.choice([
            f"垂直上升运动强烈（{w4_min:.2f} Pa/s）",
            f"强上升运动{abs(w4_min):.2f} Pa/s",
            f"显著上升运动{abs(w4_min):.2f} Pa/s"
        ])
        dynamic_parts.append(w_desc)
    elif w4_min < -0.2:
        dynamic_parts.append(f"存在上升运动（{w4_min:.2f} Pa/s）")
    
    if w4_upward_ratio > 0.3:
        dynamic_parts.append(f"上升运动区广泛（{w4_upward_ratio*100:.1f}%）")
    elif w4_upward_ratio > 0.1:
        dynamic_parts.append(f"部分区域上升运动（{w4_upward_ratio*100:.1f}%）")
    
    if vor_max > 5e-5:
        vor_desc = random.choice([
            f"涡度特征明显（{vor_max:.1e} 1/s）",
            f"存在涡旋系统（{vor_max:.1e} 1/s）",
            f"涡度活动{vor_max:.1e} 1/s"
        ])
        dynamic_parts.append(vor_desc)
    
    if vor_positive_ratio > 0.3:
        dynamic_parts.append(f"正涡度区广泛（{vor_positive_ratio*100:.1f}%）")
    
    if low_level_wind_speed > 5:
        wind_desc = random.choice([
            f"低层风速{low_level_wind_speed:.1f} m/s",
            f"风场条件{low_level_wind_speed:.1f} m/s",
            f"风速特征{low_level_wind_speed:.1f} m/s"
        ])
        dynamic_parts.append(wind_desc)
    
    if dynamic_parts:
        dynamic_intros = ["动力条件：", "环流特征：", "动力分析："]
        output_parts.append(random.choice(dynamic_intros) + "，".join(dynamic_parts) + "。")
    
    # 降水分析
    rain_analyses = []
    
    if is_heavy_rain:
        rain_intros = [
            f"强降水过程，3小时最大降水量{tp_max:.1f}毫米",
            f"强降雨天气，最大雨强{tp_max:.1f}毫米/3h",
            f"暴雨特征，降水量{tp_max:.1f}毫米"
        ]
        rain_desc = random.choice(rain_intros)
        
        if heavy_rain_ratio > 0.2:
            rain_desc += f"，强降水范围{heavy_rain_ratio*100:.1f}%"
        elif moderate_rain_ratio > 0.4:
            rain_desc += f"，中雨以上范围{moderate_rain_ratio*100:.1f}%"
            
    elif tp_max > 5:
        rain_desc = random.choice([
            f"有降水，3小时{tp_max:.1f}毫米",
            f"降雨过程，雨量{tp_max:.1f}毫米",
            f"降水天气，{tp_max:.1f}毫米"
        ])
    elif tp_max > 0.1:
        rain_desc = random.choice(["有弱降水", "零星降水", "小雨天气"])
    else:
        rain_desc = random.choice(["无明显降水", "无降水", "干燥天气"])
    
    output_parts.append(rain_desc + "。")
    
    # 高度场和温度梯度分析
    if abs(z5_gradient_ns) > 0.1:
        gradient_desc = random.choice([
            f"500hPa高度梯度{z5_gradient_ns:.3f} m/°lat",
            f"高空梯度{z5_gradient_ns:.3f} m/°lat",
            f"位势梯度{z5_gradient_ns:.3f} m/°lat"
        ])
        output_parts.append(gradient_desc + "。")
    
    if abs(temp_lapse_rate) > 5:
        lapse_desc = random.choice([
            f"温度递减率{temp_lapse_rate:.1f}℃/km",
            f"垂直温度梯度{temp_lapse_rate:.1f}℃/km",
            f"层结温度变化{temp_lapse_rate:.1f}℃/km"
        ])
        output_parts.append(lapse_desc + "。")
    
    # 时间特征分析
    if is_summer_afternoon:
        time_analysis = random.choice([
            "此时正值夏季午后，是对流天气的高发时段。",
            "夏季午后时段，对流发展条件有利。",
            "午后热力条件充分，有利于天气系统发展。"
        ])
        output_parts.append(time_analysis)
    
    # 综合风险评估
    risk_factors = []
    
    if cape_max > 1500:
        risk_factors.append("高对流能量")
    if tp_max > 10:
        risk_factors.append("强降水")
    if vor_max > 8e-5:
        risk_factors.append("强涡度")
    if w4_min < -0.3:
        risk_factors.append("强上升运动")
    if heavy_rain_ratio > 0.2:
        risk_factors.append("广泛强降水")
    
    risk_score = len(risk_factors)
    if risk_score >= 4:
        risk_level = "高"
        risk_color = "红色"
    elif risk_score >= 3:
        risk_level = "中高"
        risk_color = "橙色"
    elif risk_score >= 2:
        risk_level = "中"
        risk_color = "黄色"
    else:
        risk_level = "低"
        risk_color = "蓝色"
    
    risk_analysis = random.choice([
        f"综合天气风险等级：{risk_level}（{risk_color}预警）。",
        f"风险评估：{risk_level}级别。",
        f"风险等级：{risk_level}。"
    ])
    
    if risk_factors:
        risk_analysis += f" 主要风险因素：" + "、".join(risk_factors) + "。"
    
    output_parts.append(risk_analysis)
    
    # 专业建议
    suggestions = []
    
    if is_heavy_rain or is_stormy:
        storm_suggestions = [
            "建议：加强监测预警，做好应急准备。",
            "防范建议：注意强降水引发的次生灾害。",
            "应对措施：加强巡查，做好排水准备。"
        ]
        suggestions.append(random.choice(storm_suggestions))
    
    if is_hot:
        hot_suggestions = [
            "建议：做好电力调度和防暑降温工作。",
            "高温防范：减少户外活动，注意补水。",
            "应对措施：加强高温天气的公共服务保障。"
        ]
        suggestions.append(random.choice(hot_suggestions))
    
    if cape_max > 1000 and not is_heavy_rain:
        cape_suggestions = [
            "注意：对流潜势较高，需关注天气变化。",
            "提醒：不稳定能量积累，可能触发对流。",
            "建议：关注局地对流天气发展。"
        ]
        suggestions.append(random.choice(cape_suggestions))
    
    if suggestions:
        output_parts.append(random.choice(suggestions))
    
    # 随机添加专业总结
    summaries = [
        "以上分析基于实时气象数据，仅供参考。",
        "本分析综合了多种气象要素，反映了当前天气状况。",
        "气象条件复杂多变，请关注最新预报信息。",
        "专业气象分析，为决策提供参考依据。"
    ]
    
    if random.random() > 0.5:
        output_parts.append(random.choice(summaries))
    
    output_text = "\n".join(output_parts)
    return input_text, output_text

def get_time_info_from_numpy_datetime64(time_val):
    """从numpy.datetime64对象中提取时间信息"""
    # 转换为datetime对象
    dt = pd.Timestamp(time_val).to_pydatetime()
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    
    # 月份中文名称
    month_cn_list = ["一月", "二月", "三月", "四月", "五月", "六月", 
                     "七月", "八月", "九月", "十月", "十一月", "十二月"]
    month_cn = month_cn_list[month - 1] if 1 <= month <= 12 else f"{month}月"
    
    # 时间描述
    if 0 <= hour < 6:
        time_desc = "凌晨"
    elif 6 <= hour < 9:
        time_desc = "早晨"
    elif 9 <= hour < 12:
        time_desc = "上午"
    elif 12 <= hour < 14:
        time_desc = "中午"
    elif 14 <= hour < 18:
        time_desc = "下午"
    elif 18 <= hour < 24:
        time_desc = "夜间"
    else:
        time_desc = ""
    
    return {
        'year': year,
        'month_cn': month_cn,
        'day': day,
        'time_desc': time_desc,
        'hour': hour
    }

def generate_weather_broadcast(features):
    """生成天气播报文本"""
    if features is None:
        return "数据提取失败，无法生成播报。"
    
    # 获取时间信息
    time_str = features.get('time', '')
    try:
        time_val = np.datetime64(time_str)
        time_info = get_time_info_from_numpy_datetime64(time_val)
    except:
        time_info = {
            'year': 2026,  # 默认年份
            'month_cn': '未知',
            'day': 1,
            'time_desc': '未知',
            'hour': 0
        }
    
    # 提取关键特征值
    t2m_mean = features.get('t2m_mean', 20.0)
    t2m_max = features.get('t2m_max', 25.0)
    t2m_min = features.get('t2m_min', 15.0)
    cape_max = features.get('cape_max', 0.0)
    tp_max = features.get('tp_max', 0.0)
    w4_min = features.get('w4_min', 0.0)
    q_low_mean = features.get('q_low_mean', 0.005)
    vor_max = features.get('vor_max', 0.0)
    
    # 计算判断条件
    is_heavy_rain = tp_max > 10
    is_stormy = cape_max > 1000 or vor_max > 1e-4
    is_hot = t2m_max > 35
    is_cold = t2m_min < 0
    
    # 生成专业风格播报
    input_text = f"气象数据分析请求 - {time_info['year']}年{time_info['month_cn']}{time_info['day']}号{time_info['time_desc']}{time_info['hour']}时"
    
    _, broadcast_text = _generate_professional_style(
        input_text, features,
        time_info['year'], time_info['month_cn'], time_info['day'], 
        time_info['time_desc'], time_info['hour'],
        t2m_mean, t2m_max, t2m_min, cape_max, tp_max, w4_min,
        q_low_mean, vor_max, is_heavy_rain, is_stormy, is_hot, is_cold
    )
    
    return broadcast_text

def features_to_markdown_table(features):
    """将特征字典转换为Markdown表格"""
    if not features:
        return "无可用特征数据"
    
    # 对特征进行分类
    categories = {
        '时间信息': ['time', 'month', 'hour', 'season', 'is_summer_afternoon'],
        '温度特征': ['t2m_mean', 't2m_max', 't2m_min', 't4_mean', 't7_mean', 
                    't8_mean', 't9_mean', 'temp_lapse_rate'],
        '对流能量': ['cape_max', 'cape_mean', 'cape_high_ratio', 'stability_index'],
        '湿度特征': ['q7_mean', 'q7_max', 'q8_mean', 'q_low_mean', 'moisture_index'],
        '风场特征': ['u2_mean', 'u2_max', 'v7_mean', 'v7_max', 'low_level_wind_speed'],
        '垂直运动': ['w4_min', 'w4_mean', 'w4_upward_ratio', 'vv9_min', 'vv9_mean'],
        '涡度特征': ['vor_max', 'vor_min', 'vor_mean', 'vor_positive_ratio'],
        '水汽输送': ['vidmf_min', 'vidmf_mean', 'vidmf_convergence_ratio'],
        '位势高度': ['z2_mean', 'z2_max', 'z2_min', 'z4_mean', 'z4_max', 
                    'z4_min', 'z5_mean', 'z5_max', 'z5_min', 'z5_gradient_ns'],
        '降水特征': ['tp_max', 'tp_mean', 'heavy_rain_ratio', 'moderate_rain_ratio'],
        '地理位置': ['lat_min', 'lat_max', 'lat_mean', 'lon_min', 'lon_max', 'lon_mean']
    }
    
    markdown_lines = []
    
    for category, feature_list in categories.items():
        # 过滤出当前类别中实际存在的特征
        category_features = []
        for feature_name in feature_list:
            if feature_name in features:
                value = features[feature_name]
                # 格式化数值
                if isinstance(value, float):
                    if abs(value) < 0.001 and value != 0:
                        formatted_value = f"{value:.2e}"
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                # 添加单位说明
                unit = ""
                if 'cape' in feature_name:
                    unit = " (J/kg)"
                elif feature_name.startswith('t') and feature_name != 'time':
                    unit = " (°C)"
                elif feature_name.startswith('q'):
                    unit = " (kg/kg)"
                elif feature_name.startswith('tp'):
                    unit = " (mm/3h)"
                elif 'w4' in feature_name or 'vv' in feature_name:
                    unit = " (Pa/s)"
                elif feature_name.startswith('vor'):
                    unit = " (1/s)"
                elif feature_name.startswith('vidmf'):
                    unit = " (kg/(m²·s))"
                elif feature_name.startswith('u') or feature_name.startswith('v') or 'wind' in feature_name:
                    unit = " (m/s)"
                elif feature_name.startswith('z'):
                    unit = " (m²/s²)"
                elif 'ratio' in feature_name or '_ratio' in feature_name:
                    formatted_value = f"{value*100:.2f}%"
                
                category_features.append(f"| {feature_name} | {formatted_value}{unit} |")
        
        if category_features:
            markdown_lines.append(f"### {category}")
            markdown_lines.append("| 特征名称 | 数值 |")
            markdown_lines.append("|----------|------|")
            markdown_lines.extend(category_features)
            markdown_lines.append("")  # 空行分隔
    
    return "\n".join(markdown_lines)

def process_nc_file_to_md(nc_file_path, output_md_path, max_timesteps=None):
    """
    处理.nc文件并生成Markdown文档
    
    参数:
    ----------
    nc_file_path : str
        .nc文件路径
    output_md_path : str
        输出.md文件路径
    max_timesteps : int, optional
        最大处理时次数量，None表示处理所有时次
    """
    print(f"正在处理文件: {nc_file_path}")
    
    try:
        # 1. 打开nc文件
        ds = xr.open_dataset(nc_file_path)
        print(f"文件打开成功，变量: {list(ds.data_vars)}")
        
        # 2. 检查时间维度
        if 'time' not in ds.dims:
            print("警告：未找到'time'维度，将处理整个数据集作为单个时次")
            time_coords = [None]
        else:
            time_coords = ds['time'].values
            print(f"找到{len(time_coords)}个时次")
        
        # 3. 限制处理的时次数量
        if max_timesteps is not None and len(time_coords) > max_timesteps:
            time_coords = time_coords[:max_timesteps]
            print(f"将处理前{max_timesteps}个时次")
        
        # 4. 准备Markdown内容
        md_content = [
            "# 气象数据分析报告",
            "",
            f"**数据文件**: {Path(nc_file_path).name}",
            f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**总时次数**: {len(time_coords)}",
            "",
            "---",
            ""
        ]
        
        processed_count = 0
        error_count = 0
        
        # 5. 遍历每个时次
        for i, time_coord in enumerate(time_coords):
            try:
                # 提取当前时次的数据切片
                if time_coord is None:
                    data_slice = ds
                    time_str = "无时间信息"
                else:
                    data_slice = ds.sel(time=time_coord)
                    time_str = str(time_coord)
                
                print(f"处理时次 {i+1}/{len(time_coords)}: {time_str}")
                
                # 6. 提取特征
                features = extract_features_single_timestep(data_slice)
                
                if features is None:
                    print(f"  警告：时次 {i+1} 特征提取失败")
                    error_count += 1
                    continue
                
                # 7. 生成天气播报
                broadcast = generate_weather_broadcast(features)
                
                # 8. 将特征转换为Markdown表格
                features_table = features_to_markdown_table(features)
                
                # 9. 添加到Markdown内容
                time_info = ""
                if 'time' in features:
                    time_info = f"**时间**: {features['time']}"
                
                md_content.append(f"## 时次 {i+1}")
                md_content.append("")
                md_content.append(time_info)
                md_content.append("")
                md_content.append("### 专业气象播报")
                md_content.append("")
                # 将播报文本拆分为多行，每行作为一个段落
                broadcast_lines = broadcast.split('\n')
                for line in broadcast_lines:
                    md_content.append(f"{line}")
                md_content.append("")
                md_content.append("### 详细特征数据")
                md_content.append("")
                md_content.append(features_table)
                md_content.append("")
                md_content.append("---")
                md_content.append("")
                
                processed_count += 1
                
            except Exception as e:
                print(f"  处理时次 {i+1} 时出错: {e}")
                error_count += 1
                continue
        
        # 11. 写入Markdown文件
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_content))
        
        print(f"处理完成！")
        print(f"成功处理: {processed_count} 个时次")
        print(f"处理失败: {error_count} 个时次")
        print(f"Markdown文件已保存至: {output_md_path}")
        
        # 12. 关闭数据集
        ds.close()
        
        return {
            'success': True,
            'output_path': output_md_path,
            'processed_count': processed_count,
            'error_count': error_count,
            'total_timesteps': len(time_coords)
        }
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }
if __name__ == "__main__":
    nc_file_path = "/home/guojn/code/qikeyuan/DATA/NEWDATA_3h/ERA5_hourly_output/ERA5_CIRCUL_1961_2021_3H_025x025_18.nc"  # 替换为你的.nc文件路径
    output_md_path = "weather_rag.md"
    
    # 处理文件 ，max_timestpes可酌情增加/减小
    reslut = process_nc_file_to_md(nc_file_path, output_md_path, max_timesteps)

# %%
# 以下是再对生成的.md文件进行一次整理
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, asdict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MeteorologicalData:
    """气象数据结构体"""
    timestep: str
    time: str
    summary: str
    metadata: Dict[str, Any]
    detailed_data: Dict[str, Dict[str, Any]]
    raw_content: str
    file_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    def to_structured_dict(self) -> Dict[str, Any]:
        """转换为RAG友好的结构化字典"""
        return {
            "id": self.timestep,
            "metadata": self.metadata,
            "summary": self.summary,
            "detailed_data": self.detailed_data,
            "source": self.file_path
        }

class MeteorologicalFileParser:
    """气象Markdown文件解析器"""
    
    def __init__(self, chunks_dir: str):
        self.chunks_dir = Path(chunks_dir)
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"目录不存在: {chunks_dir}")
    
    def parse_single_file(self, file_path: Path) -> Optional[MeteorologicalData]:
        """解析单个Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取时次
            timestep_match = re.search(r'## 时次\s+(\d+)', content)
            if not timestep_match:
                logger.warning(f"文件 {file_path.name} 中未找到时次信息")
                return None
            
            timestep = timestep_match.group(1)
            
            # 提取时间
            time_match = re.search(r'\*\*时间\*\*:\s*(.+?)\s*\n', content)
            time_str = time_match.group(1).strip() if time_match else ""
            
            # 提取专业气象播报部分
            summary_match = re.search(r'### 专业气象播报\s*\n(.*?)\n### 详细特征数据', 
                                     content, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else ""
            
            # 解析详细特征数据
            detailed_data = self._parse_detailed_tables(content)
            
            # 构建元数据
            metadata = self._build_metadata(time_str, detailed_data)
            
            return MeteorologicalData(
                timestep=timestep,
                time=time_str,
                summary=summary,
                metadata=metadata,
                detailed_data=detailed_data,
                raw_content=content,
                file_path=str(file_path)
            )
            
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时出错: {e}")
            return None
    
    def _parse_detailed_tables(self, content: str) -> Dict[str, Dict[str, Any]]:
        """解析所有详细特征数据表格"""
        detailed_data = {}
        
        # 找到详细特征数据部分
        detailed_section_match = re.search(r'### 详细特征数据\s*(.*)', content, re.DOTALL)
        if not detailed_section_match:
            return detailed_data
        
        detailed_section = detailed_section_match.group(1)
        
        # 查找所有表格标题和内容
        # 格式: ### 表标题\n| 特征名称 | 数值 |\n|----------|------|\n| ... | ... |
        table_pattern = r'###\s+([^\n]+?)\s*\n\|\s*特征名称\s*\|\s*数值\s*\|\s*\n\|\s*-+\s*\|\s*-+\s*\|\s*\n([\s\S]*?)(?=\n###|\n---|\Z)'
        
        for match in re.finditer(table_pattern, detailed_section):
            table_name = match.group(1).strip()
            table_content = match.group(2)
            
            table_data = {}
            
            # 解析表格行
            row_pattern = r'\|\s*([^\|]+?)\s*\|\s*([^\|]+?)\s*\|\s*\n'
            for row_match in re.finditer(row_pattern, table_content):
                feature_name = row_match.group(1).strip()
                feature_value = row_match.group(2).strip()
                
                # 尝试从特征值中提取数值和单位
                parsed_value = self._parse_feature_value(feature_value)
                table_data[feature_name] = parsed_value
            
            if table_data:
                detailed_data[table_name] = table_data
        
        return detailed_data
    
    def _parse_feature_value(self, value_str: str) -> Dict[str, Any]:
        """解析特征值，分离数值和单位"""
        result = {
            "raw": value_str,
            "value": None,
            "unit": None,
            "numeric_value": None
        }
        
        # 尝试匹配数值和单位，例如: "20.7122 (°C)" 或 "1.71e-04 (1/s)"
        pattern = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:\(([^)]+)\))?'
        match = re.search(pattern, value_str)
        
        if match:
            numeric_part = match.group(1)
            unit_part = match.group(2) if match.group(2) else ""
            
            # 尝试转换为数值
            try:
                # 处理科学计数法
                if 'e' in numeric_part.lower():
                    result["numeric_value"] = float(numeric_part)
                # 处理百分数
                elif '%' in value_str:
                    result["numeric_value"] = float(numeric_part) / 100
                else:
                    result["numeric_value"] = float(numeric_part)
            except ValueError:
                # 如果不能转换为数值，可能是字符串（如"summer"）
                result["value"] = value_str
            
            result["unit"] = unit_part if unit_part else None
        
        # 如果没有匹配到数值模式，直接存储为字符串
        if result["numeric_value"] is None and result["value"] is None:
            result["value"] = value_str
        
        return result
    
    def _build_metadata(self, time_str: str, detailed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """构建元数据"""
        metadata = {
            "time": time_str
        }
        
        # 从详细数据中提取关键元数据
        if "时间信息" in detailed_data:
            time_info = detailed_data["时间信息"]
            for key, value_dict in time_info.items():
                if value_dict["value"] is not None:
                    metadata[key] = value_dict["value"]
                elif value_dict["numeric_value"] is not None:
                    metadata[key] = value_dict["numeric_value"]
        
        # 尝试解析时间字符串为datetime对象
        try:
            # 移除可能的纳秒部分
            clean_time = re.sub(r'\.\d+', '', time_str)
            dt = datetime.fromisoformat(clean_time.replace('Z', '+00:00'))
            metadata["datetime"] = dt.isoformat()
            metadata["year"] = dt.year
            metadata["month"] = dt.month
            metadata["day"] = dt.day
            metadata["hour"] = dt.hour
        except Exception as e:
            logger.debug(f"无法解析时间字符串 {time_str}: {e}")
        
        return metadata
    
    def parse_all_files(self) -> List[MeteorologicalData]:
        """解析目录下的所有文件"""
        all_data = []
        md_files = list(self.chunks_dir.glob("*.md"))
        
        logger.info(f"开始解析 {len(md_files)} 个文件...")
        
        for i, file_path in enumerate(md_files, 1):
            parsed_data = self.parse_single_file(file_path)
            if parsed_data:
                all_data.append(parsed_data)
            
            if i % 100 == 0:
                logger.info(f"已解析 {i}/{len(md_files)} 个文件")
        
        logger.info(f"解析完成，成功解析 {len(all_data)} 个文件")
        return all_data

class DataProcessor:
    """数据处理和保存类"""
    
    def __init__(self, output_dir: str = "./processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_as_json(self, data_list: List[MeteorologicalData], filename: str = "meteorological_data.json"):
        """保存为JSON文件"""
        output_path = self.output_dir / filename
        
        # 转换为结构化字典
        structured_data = []
        for data in data_list:
            structured_data.append(data.to_structured_dict())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到 {output_path}")
        return output_path
    
   

def main():
    """主函数"""
    # 配置路径
    chunks_dir = "/home/guojn/code/qikeyuan/DATA/timestep_chunks" #替换成你的.md文件夹路径
    output_dir = "./processed_meteorological_data" #生成.json文件的路径
    
    # 1. 解析所有文件
    parser = MeteorologicalFileParser(chunks_dir)
    all_data = parser.parse_all_files()
    
    if not all_data:
        logger.error("没有解析到任何数据")
        return
    
    # 2. 处理数据
    processor = DataProcessor(output_dir)
    
    # 3. 保存为JSON
    json_path = processor.save_as_json(all_data)
   
if __name__ == "__main__":
    main()

# 如果顺利的话，此时已经生成了一个.json文件，也就是该目录下的meteorological_data.json
