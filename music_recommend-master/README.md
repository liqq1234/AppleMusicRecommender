# 毕业设计--基于Django 5.1的歌曲推荐系统

## 说明 ✨

该项目已于 2026 年完成现代化升级，全面支持 **Python 3.11 - 3.13**。解决了旧版本中 Pillow 无法在 Windows 安装、Django 版本过低等历史遗留问题。

1. **开发建议**：推荐使用 PyCharm (Professional版) 结合虚拟环境进行开发。
2. **管理员账号**：通过 `python manage.py createsuperuser` 创建。
3. **数据管理**：
    *   音乐导入：运行 `scripts/populate_musics_script.py` (会清空现有音乐并重新从 CSV 导入)。
    *   用户模拟：运行 `scripts/populate_user_rate.py` 生成模拟评分数据。

## 系统架构与技术栈 🚀

*   **后端**: Django 5.1.2 (MVC 架构)
*   **数据库**: SQLite3 (轻量级，无需配置环境)
*   **推荐算法**: 
    *   **基于用户 (User-CF)**: 采用 Pearson 相关系数计算用户相似度。
    *   **基于项目 (Item-CF)**: 利用余弦相似度进行音乐关联预测。
    *   **矩阵分解 (SVD)**: 核心推荐引擎，由 Scikit-learn 驱动。
*   **数据流**: Python 爬虫采集 -> CSV 暂存 -> Django ORM 入库。

## 安装与运行方法 (新电脑快速部署) 🛠️

### 1. 准备环境
确保电脑已安装 Python 3.11 或更高版本。

### 2. 克隆与安装依赖
```powershell
# 进入项目目录
cd music_recommend-master

# 安装所需的所有库 (包含 Django, Scikit-learn, Pandas 等)
pip install -r requirements.txt --user
```

### 3. 初始化数据库
```powershell
python manage.py makemigrations
python manage.py migrate
```

### 4. 导入初始数据 (如果数据库为空)
```powershell
# 导入音乐
python scripts/populate_musics_script.py
# 导入评分
python scripts/populate_user_rate.py
```

### 5. 启动系统
```powershell
python manage.py runserver
```
访问地址: `http://127.0.0.1:8000/`

## 各目录功能解析 📂

*   `core/`: 项目核心配置，包含 `settings.py` (配置文件) 和 `urls.py` (全局路由)。
*   `user/`: 主应用模块。
    *   `models.py`: 数据库模型（歌曲、用户、评分、标签）。
    *   `views.py`: 核心业务逻辑实现（推荐接口、页面渲染）。
    *   `recommend_engine.py`: 推荐算法底层逻辑 (SVD, User-CF, Item-CF)。
*   `data/`: 存放 `db.sqlite3` 数据库和原始 `cloudmusic.csv` 数据。
*   `scripts/`: 数据导入和初始化脚本。
*   `static/`: 存放 CSS、JS 以及 UI 资源。


