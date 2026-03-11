import os
import django

# 设置 Django 环境
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from user.recommend_engine import evaluate_all_models

def main():
    print("="*60)
    print("   推荐系统算法性能多维对比评估 (学术论文版)")
    print("="*60)
    
    try:
        all_metrics = evaluate_all_models()
        
        if "error" in all_metrics:
            print(f"提示: {all_metrics['error']}")
            return

        # 打印表头
        header = f"{'算法模型':<18} | {'RMSE':<8} | {'MAE':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<8} | {'Coverage':<8}"
        print(header)
        print("-" * len(header))
        
        for name, m in all_metrics.items():
            print(f"{name:<16} | {m['RMSE']:<8} | {m['MAE']:<8} | {m['Precision']*100:>8.2f}% | {m['Recall']*100:>7.2f}% | {m['F1']:<8} | {m['Coverage']*100:>7.2f}%")
        
        print("\n指标定义 (便于论文引用):")
        print("1. RMSE/MAE: 回归误差。反映分值预测的精准度，越低越好。")
        print("2. Precision/Recall/F1: 分类指标。反映 Top-N 推荐的质量，越高越好。")
        print("3. Coverage (覆盖率): 能够推荐出的歌曲占总曲库的比例，反映模型对长尾歌曲的挖掘能力。")
        print("\n* 结论建议：通常 SVD 在稀疏数据下表现最优，而 Item-CF 在实时性上更有优势。")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
