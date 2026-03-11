import os
import django

# 设置 Django 环境（以便在单独的脚本或模块中安全使用 Django 的 ORM）
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from user.models import Music, Rate, User
from user.recommend_engine import recommend_for_user_svd
def recommend_by_user_id(user_id):
    """
    基于用户的 SVD 矩阵分解算法推荐 (Scikit-learn TruncatedSVD)
    提取了隐含特征（如曲风、歌手风格）并在计算中考虑了播放时长等隐式偏好。
    """
    try:
        # 调用新写的机器学习算法核心模块
        musics = recommend_for_user_svd(user_id, top_n=10)
        
        # 冷启动处理：如果给一个没有数据的全新用户，SVD可能无法提供好的推荐
        # 此时降级(Fallback)到流行度推荐（这里是收藏人数最多的）
        if not musics:
            musics = list(Music.objects.order_by('-sump')[:10])
            
        return musics
    except Exception as e:
        print(f"Error in recommend_by_user_id (SVD): {e}")
        # 如果算法出问题，保底返回浏览量大的数据
        return list(Music.objects.order_by('-num')[:10])

def recommend_by_item_id(user_id):
    """
    基于物品的协同过滤推荐 (Item-CF)
    目前暂时返回收藏数最高的歌曲作为占位。
    """
    try:
        # 简单占位逻辑：返回收藏量最高的 10 首歌曲
        musics = Music.objects.order_by('-sump')[:10]
        return list(musics)
    except Exception as e:
        print(f"Error in recommend_by_item_id: {e}")
        return []
