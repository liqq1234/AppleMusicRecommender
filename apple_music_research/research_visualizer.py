import os
import django
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Windows 下的字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 初始化 Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from user.recommend_engine import evaluate_all_models, get_data_df

def plot_performance_comparison(metrics_dict):
    """
    绘制不同算法的性能指标对比图
    """
    # 转换为 DataFrame 方便 Seaborn 绘图
    data = []
    for model_name, metrics in metrics_dict.items():
        for metric_name, val in metrics.items():
            if metric_name in ['RMSE', 'MAE']: # 误差指标
                data.append({'Model': model_name, 'Metric': metric_name, 'Value': val, 'Type': 'Error'})
            elif metric_name in ['Precision', 'Recall', 'F1', 'Coverage', 'Diversity']: # 质量指标
                data.append({'Model': model_name, 'Metric': metric_name, 'Value': val, 'Type': 'Accuracy/Quality/Diversity'})
    
    df_plot = pd.DataFrame(data)
    
    # 1. 误差对比 (RMSE/MAE) - 越低越好
    plt.figure(figsize=(10, 6))
    error_data = df_plot[df_plot['Type'] == 'Error']
    sns.barplot(x='Metric', y='Value', hue='Model', data=error_data, palette='viridis')
    plt.title('推荐算法误差指标对比 (RMSE/MAE) - 越低越好', fontsize=14)
    plt.ylabel('分值偏差')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('error_comparison.png', dpi=300)
    print("已保存: error_comparison.png")

    # 2. 精准度/多样性对比 - 越高越好
    plt.figure(figsize=(14, 6))
    quality_data = df_plot[df_plot['Type'] == 'Accuracy/Quality/Diversity']
    sns.barplot(x='Metric', y='Value', hue='Model', data=quality_data, palette='magma')
    plt.title('推荐算法综合性能对比 (Accuracy/Quality/Diversity) - 越高越好', fontsize=14)
    plt.ylabel('数值评分')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('quality_comparison.png', dpi=300)
    print("已保存: quality_comparison.png")

def plot_user_behavior_analysis():
    """
    绘制用户行为特征分析图 (Apple Music 特色: 播放比例/跳过次数)
    """
    df = get_data_df()
    if df.empty:
        print("无交互数据，跳过行为分析图")
        return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='play_ratio', fill=True, color='blue', label='播放比例')
    plt.title('Apple Music 用户听歌完成度(Play Ratio)分布', fontsize=14)
    plt.xlabel('播放比例 (0.0=秒切, 1.0=完整播放)')
    plt.ylabel('用户密度')
    plt.savefig('user_behavior_play_ratio.png', dpi=300)
    print("已保存: user_behavior_play_ratio.png")

    plt.figure(figsize=(10, 6))
    sns.countplot(x='mark', data=df, palette='muted')
    plt.title('用户显性打分 (Ratings) 分布', fontsize=14)
    plt.xlabel('评分 (1-5 星)')
    plt.ylabel('频次')
    plt.savefig('rating_distribution.png', dpi=300)
    print("已保存: rating_distribution.png")

def main():
    print("正在启动 Apple Music 音乐预测建模与对比分析项目...")
    
    # 1. 获取评估指标
    metrics = evaluate_all_models()
    if "error" in metrics:
        print(f"数据量不足: {metrics['error']}")
        return

    # 2. 生成图表
    print("正在生成实验结果图表...")
    plot_performance_comparison(metrics)
    plot_user_behavior_analysis()
    
    print("\n[分析项目已就绪]")
    print("生成的图表列表:")
    print("- error_comparison.png: RMSE/MAE 误差对比")
    print("- quality_comparison.png: 准确率/召回率/覆盖率对比")
    print("- user_behavior_play_ratio.png: 听歌行为特征分布")
    print("- rating_distribution.png: 基础分值分布")
    print("\n这些图表可以直接插入您的毕业论文中。")

if __name__ == "__main__":
    main()
