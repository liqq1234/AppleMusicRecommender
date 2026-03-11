import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import logging

logger = logging.getLogger(__name__)

def get_data_df():
    """
    从数据库加载交互数据，并进行特征工程（计算隐性反馈）。
    """
    from user.models import Rate
    rates = Rate.objects.all().values('user_id', 'music_id', 'mark', 'play_duration', 'total_duration', 'skip_count')
    if not rates:
        return pd.DataFrame()
        
    df = pd.DataFrame(list(rates))
    
    # 【Apple Music 核心特征工程】：引入播放比例和跳过惩罚
    # 兴趣分值 = 显性分数(mark) + (播放时长/总时长 * 2) - (跳过次数 * 1)
    df['total_duration'] = df['total_duration'].replace(0, 1) # 防止除以0
    df['play_ratio'] = df['play_duration'] / df['total_duration']
    
    # 综合计算综合兴趣分（隐性+显性）
    df['interest_score'] = df['mark'] + (df['play_ratio'] * 2.0) - (df['skip_count'] * 1.0)
    # 将分值截断在 1-5 之间，便于模型理解
    df['interest_score'] = df['interest_score'].clip(1, 5)
    
    return df

def build_svd_model():
    """
    建立 SVD 矩阵分解模型。
    提取歌曲之间的深层潜在特征。
    """
    df = get_data_df()
    if df.empty:
        return None, None
        
    # 构建 User-Item 稀疏兴趣矩阵
    ui_matrix = df.pivot_table(index='user_id', columns='music_id', values='interest_score').fillna(0)
    
    # 设置潜在因子维度（Latent Factors），上限为20维
    n_components = min(20, ui_matrix.shape[1] - 1)
    if n_components <= 0:
        return None, None
        
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    # 分解再重构，得到降维后的预测得分矩阵
    reduced_matrix = svd.fit_transform(ui_matrix)
    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    
    # 构建带索引的 DataFrame 便于查询
    predictions_df = pd.DataFrame(reconstructed_matrix, index=ui_matrix.index, columns=ui_matrix.columns)
    
    return predictions_df, df

def recommend_for_user_svd(user_id, top_n=10):
    """
    基于 SVD 模型为指定用户推荐 Top-N 音乐。
    """
    from user.models import Music
    predictions_df, df = build_svd_model()
    
    if predictions_df is None or user_id not in predictions_df.index:
        return []
        
    # 获取目标用户的所有的预测打分
    user_predictions = predictions_df.loc[user_id].sort_values(ascending=False)
    
    # 获取用户已经产生过交互行为的音乐，避免重复推荐
    interacted_musics = df[df['user_id'] == user_id]['music_id'].tolist()
    
    # 过滤掉已听过的音乐
    recommendations_series = user_predictions[~user_predictions.index.isin(interacted_musics)]
    
    # 取前 N 首歌曲的 ID
    top_music_ids = recommendations_series.head(top_n).index.tolist()
    
    # 从数据库获取 Music 实体并保持原推荐顺序
    musics = []
    for mid in top_music_ids:
        try:
            musics.append(Music.objects.get(id=mid))
        except Exception as e:
            logger.warning(f"Music ID {mid} not found: {e}")
            continue
            
    return musics

def build_user_cf_model(train_matrix):
    """
    基于用户的协同过滤 (User-Based CF)
    使用皮尔逊相关系数计算用户相似度
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 计算用户相似度矩阵
    user_sim_matrix = cosine_similarity(train_matrix)
    user_sim_df = pd.DataFrame(user_sim_matrix, index=train_matrix.index, columns=train_matrix.index)
    
    # 预测得分：相似用户加权平均
    # 为了简化计算，我们直接使用矩阵乘法
    mean_user_rating = train_matrix.mean(axis=1)
    ratings_diff = (train_matrix.T - mean_user_rating).T
    
    # 避免除以 0
    sim_sum = np.array([np.abs(user_sim_matrix).sum(axis=1)]).T
    sim_sum[sim_sum == 0] = 1.0
    
    pred = mean_user_rating.values.reshape(-1, 1) + user_sim_matrix.dot(ratings_diff) / sim_sum
    
    predictions_df = pd.DataFrame(pred, index=train_matrix.index, columns=train_matrix.columns).fillna(0)
    return predictions_df

def build_item_cf_model(train_matrix):
    """
    基于物品的协同过滤 (Item-Based CF)
    使用余弦相似度计算物品相似度
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 计算物品相似度矩阵 (Item-Item)
    item_sim_matrix = cosine_similarity(train_matrix.T)
    
    # 避免除以 0
    sim_sum = np.array([np.abs(item_sim_matrix).sum(axis=1)])
    sim_sum[sim_sum == 0] = 1.0
    
    # 预测得分：用户已评分物品的加权平均
    pred = train_matrix.dot(item_sim_matrix) / sim_sum
    
    predictions_df = pd.DataFrame(pred, index=train_matrix.index, columns=train_matrix.columns).fillna(0)
    return predictions_df

def calculate_metrics(y_true, y_pred, threshold=3.5):
    """
    计算学术论文要求的核心指标
    """
    if not y_true:
        return {
            "RMSE": 0, "MAE": 0, "Precision": 0, "Recall": 0, "F1": 0
        }
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 回归指标
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 分类指标 (Top-N 推荐视角)
    true_positives = 0
    predicted_positives = 0
    actual_positives = 0
    
    for t, p in zip(y_true, y_pred):
        if p >= threshold:
            predicted_positives += 1
            if t >= threshold:
                true_positives += 1
        if t >= threshold:
            actual_positives += 1
            
    precision = (true_positives / predicted_positives) if predicted_positives > 0 else 0.0
    recall = (true_positives / actual_positives) if actual_positives > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4)
    }

def evaluate_all_models():
    """
    全量指标对比评估：SVD vs User-CF vs Item-CF
    为论文提供全套数据。
    """
    df = get_data_df()
    if df.empty or len(df) < 20:
        return {"error": "数据量不足，无法进行学术评估对比"}
        
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # 统一构建基础矩阵
    train_matrix = train_data.pivot_table(index='user_id', columns='music_id', values='interest_score').fillna(0)
    
    # 1. SVD 评估
    n_components = min(20, train_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(train_matrix)
    reconstructed = svd.inverse_transform(reduced)
    svd_preds = pd.DataFrame(reconstructed, index=train_matrix.index, columns=train_matrix.columns)
    
    # 2. User-CF 评估
    user_cf_preds = build_user_cf_model(train_matrix)
    
    # 3. Item-CF 评估
    item_cf_preds = build_item_cf_model(train_matrix)
    
    results = {}
    models = {
        "SVD (矩阵分解)": svd_preds,
        "User-CF (基于用户)": user_cf_preds,
        "Item-CF (基于物品)": item_cf_preds
    }
    
    for name, pred_df in models.items():
        y_true = []
        y_pred = []
        for _, row in test_data.iterrows():
            u, i, score = row['user_id'], row['music_id'], row['interest_score']
            if u in pred_df.index and i in pred_df.columns:
                y_true.append(score)
                y_pred.append(pred_df.loc[u, i])
        
        results[name] = calculate_metrics(y_true, y_pred)
        
        # 计算覆盖率 (Coverage): 推荐出的独特歌曲占总歌曲的比例
        # 简单模拟：预测分 > 3.5 的歌曲 ID 集合
        recommended_items = set()
        for u in pred_df.index:
            top_items = pred_df.loc[u].sort_values(ascending=False).head(10)
            recommended_items.update(top_items[top_items >= 3.5].index.tolist())
        
        total_items = df['music_id'].nunique()
        results[name]["Coverage"] = round(len(recommended_items) / total_items, 4) if total_items > 0 else 0

    return results

def evaluate_model():
    """
    兼容性封装：返回默认算法（SVD）的精简指标
    """
    res = evaluate_all_models()
    if "error" in res:
        return {"RMSE": 0, "Precision": 0}
    return res["SVD (矩阵分解)"]
