import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import pickle
import os
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_data_df():
    """统一数据入口，加载交互记录并预处理"""
    from user.models import Rate
    from django.utils import timezone
    
    rates = Rate.objects.all().values(
        'user_id', 'music_id', 'mark',
        'play_duration', 'total_duration', 'skip_count', 'create_time'
    )
    if not rates:
        return pd.DataFrame()
        
    df = pd.DataFrame(list(rates))
    
    # 基本清洗
    df['total_duration'] = df['total_duration'].replace(0, 1)
    df['play_ratio'] = (df['play_duration'] / df['total_duration']).clip(0, 1)
    
    # 兴趣建模 (包含时间衰减)
    now = timezone.now()
    df['create_time'] = pd.to_datetime(df['create_time'])
    days_passed = (now - df['create_time']).dt.total_seconds() / 86400
    decay_weight = np.exp(-0.02 * days_passed) # 适度减缓衰减
    
    base_score = df['mark'] + df['play_ratio'] * 2.0 - df['skip_count'] * 1.0
    df['interest_score'] = (base_score * decay_weight).clip(1, 5)
    
    return df

def get_sparse_matrix(df):
    """构建稀疏矩阵以节省内存，支持 10万+ 维维度"""
    # 映射 ID 到 0-based 索引
    user_ids = df['user_id'].astype('category')
    music_ids = df['music_id'].astype('category')
    
    matrix = csr_matrix((df['interest_score'], 
                        (user_ids.cat.codes, music_ids.cat.codes)))
    
    return matrix, user_ids.cat.categories, music_ids.cat.categories

def build_svd_model(force_recompute=False):
    """带缓存的 SVD 实现，针对 106k 数据优化"""
    cache_path = os.path.join(CACHE_DIR, 'svd_model.pkl')
    
    if not force_recompute and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
            
    df = get_data_df()
    if df.empty or len(df) < 10:
        return None, None, None, None
        
    sparse_mat, u_cats, m_cats = get_sparse_matrix(df)
    
    # SVD 降维计算
    n_components = min(15, sparse_mat.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    # 得到降维后的用户特征与物品特征
    user_features = svd.fit_transform(sparse_mat)
    item_features = svd.components_.T
    
    model_data = (user_features, item_features, u_cats, m_cats)
    with open(cache_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    return model_data

def recommend_for_user_svd(user_id, top_n=10):
    """百万级数据秒级推荐接口"""
    from user.models import Music
    model_data = build_svd_model()
    
    if model_data is None:
        return Music.objects.all().order_by('-num')[:top_n]
        
    user_features, item_features, u_cats, m_cats = model_data
    
    if user_id not in u_cats:
        return Music.objects.all().order_by('-num')[:top_n]
        
    # 1. 找到用户所在的行索引
    u_idx = u_cats.get_loc(user_id)
    u_feat = user_features[u_idx]
    
    # 2. 通过向量点积计算该用户对所有物品的预测分数 (极其高效)
    scores = u_feat.dot(item_features.T)
    
    # 3. 排序并过滤已听过的
    # 获取该用户已听过的 music_id
    df = get_data_df() # 仅用于过滤，后续可进一步优化
    interacted_ids = set(df[df['user_id'] == user_id]['music_id'])
    
    pred_series = pd.Series(scores, index=m_cats)
    recommendations = pred_series[~pred_series.index.isin(interacted_ids)].sort_values(ascending=False)
    
    # 【改动】为了支持“换一批”，我们从候选池（如 Top-100）中随机采样
    pool_size = max(top_n * 5, 100)
    candidate_ids = recommendations.head(pool_size).index.tolist()
    
    import random
    if len(candidate_ids) > top_n:
        top_ids = random.sample(candidate_ids, top_n)
    else:
        top_ids = candidate_ids
    
    # 4. 获取数据库对象
    musics = []
    music_map = {m.id: m for m in Music.objects.filter(id__in=top_ids)}
    for mid in top_ids: # 保持采样后的顺序（可选，也可打乱）
        if mid in music_map:
            musics.append(music_map[mid])
            
    return musics
