import os
import django
import pandas as pd
import numpy as np

# 初始化 Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from user.recommend_engine import evaluate_all_models

def run_benchmark():
    """
    运行全量算法基准测试并输出学术报表
    """
    print("开始算法性能全量基准测试...")
    results = evaluate_all_models()
    
    if "error" in results:
        print(f"评估失败: {results['error']}")
        return
    
    # 构建学术统计表
    report_data = []
    for model_name, metrics in results.items():
        row = {"算法模型": model_name}
        row.update(metrics)
        report_data.append(row)
    
    df_report = pd.DataFrame(report_data)
    
    print("\n" + "="*80)
    print("   推荐系统算法性能多维对比评估报告 (学术论文版)")
    print("="*80)
    
    # 格式化输出表格
    print(df_report.to_string(index=False))
    
    print("\n指标解读:")
    print("1. RMSE/MAE: 反映分值预测偏差，越小越好。")
    print("2. Precision/Recall/F1: 反映 Top-N 推荐命中率，越大越好。")
    print("3. Coverage: 反映模型对长尾曲库的覆盖能力（发现冷门好歌的能力）。")
    print("4. Diversity: 反映推荐列表风格的丰富度（信息熵），越大越好。")
    print("="*80)
    
    return df_report

if __name__ == "__main__":
    run_benchmark()
