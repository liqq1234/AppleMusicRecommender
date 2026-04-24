import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from .evaluator import evaluate_all_models
from .recommend_engine import get_data_df

def generate_academic_charts(output_dir='figures'):
    """生成面向学术论文的标准化图表"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    metrics = evaluate_all_models()
    if "error" in metrics:
        return metrics["error"]
        
    # 转换为 DataFrame 方便可视化
    results_list = []
    for algo, m in metrics.items():
        m['Algorithm'] = algo
        results_list.append(m)
    res_df = pd.DataFrame(results_list)
    
    # 1. 算法误差对比图 (RMSE/MAE)
    plt.figure(figsize=(10, 6))
    melted_err = res_df.melt(id_vars='Algorithm', value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Value')
    sns.barplot(data=melted_err, x='Metric', y='Value', hue='Algorithm', palette='viridis')
    plt.title('Error Comparison (RMSE & MAE)')
    plt.savefig(os.path.join(output_dir, 'error_comparison.png'))
    plt.close()
    
    # 2. 综合性能对比图 (Precision, Recall, F1, Coverage)
    plt.figure(figsize=(12, 7))
    melted_qual = res_df.melt(id_vars='Algorithm', value_vars=['Precision', 'Recall', 'F1', 'Coverage'], var_name='Metric', value_name='Value')
    sns.barplot(data=melted_qual, x='Metric', y='Value', hue='Algorithm', palette='magma')
    plt.title('Quality Comparison (P/R/F1/Coverage)')
    plt.savefig(os.path.join(output_dir, 'quality_comparison.png'))
    plt.close()
    
    # 3. 用户行为特征分布图 (Play Ratio)
    df = get_data_df()
    if not df.empty:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df['play_ratio'], fill=True, color='red')
        plt.title('User Behavior Distribution (Play Completion Ratio)')
        plt.xlabel('Play Ratio')
        plt.savefig(os.path.join(output_dir, 'user_behavior_play_ratio.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='mark', palette='Reds')
        plt.title('Explicit Rating Distribution')
        plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
        plt.close()
        
    return "Charts generated successfully in " + output_dir
