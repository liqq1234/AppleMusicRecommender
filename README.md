# Apple Music 推荐系统与数据分析项目

欢迎使用本基于协同过滤的 Apple Music 音乐推荐及数据分析平台项目。本项目分为两个主要部分：

1. **`music_recommend-master`**: 基于 Django 构建的音乐推荐系统，包含完整的前端用户交互展示、后端服务逻辑以及内部自带的 SQLite 数据库存储方案。
2. **`apple_music_research`**: 用于支撑学术论文中推荐算法性能对比、模型可视化与模拟实验的科研模块。

本文档将详细指导您如何从零开始部署并运行该项目。

---

## 🚀 一、 系统依赖与环境配置

在运行本项目前，建议您确认电脑中已安装以下环境：
- **Python 环境**: 推荐使用 Python 3.7 - 3.9 版本。
- **数据库**: 系统自带轻量级的 SQLite3 数据库，**无需**额外安装 MySQL 或其他数据库软件。
- **前端工具**: 项目使用了静态的 Bootstrap3 框架，**无需**使用 npm 安装 Node.js 等依赖，即开即用。

---

## 🛠 二、 推荐系统核心启动教程 (music_recommend-master)

进入核心操作目录：
```bash
cd music_recommend-master
```

### 1. 安装项目必要依赖包
推荐您在虚拟环境中执行该安装命令以防与其他项目冲突。
```bash
pip install -r requirements.txt
```
*(提示：如果下载速度过慢，请尝试在命令后补充国内镜像源参数，例如 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`)*

### 2. 数据库迁移与初始化
如果您是首次运行本项目，您需要执行以下命令同步数据库架构内容：
```bash
python manage.py makemigrations
python manage.py migrate
```

### 3. 数据爬取与虚拟评分填充 (重要!)
我们提供了一些脚本可一键填补数据到数据库内以供模拟测试：
```bash
# 执行爬虫程序自动向数据库中插入歌曲预设数据 (警告：会覆盖部分原有数据！)
python populate_movies_script.py

# 为系统随机生成虚拟的用户评分，这一步是协同过滤(推荐系统)能运作的基础
python populate_user_rate.py
```

### 4. 设置开发者后台管理员
为了能够登录后台管理整个论坛或音乐列表，必须建立一个超级管理员：
```bash
python manage.py createsuperuser
# 此时根据终端提示：逐步输入您希望设置的 管理员名称、联系邮箱 以及 管理员密码。
```

### 5. 启动项目运行
执行以下命令启动 Django 的服务器环境：
```bash
python manage.py runserver 0.0.0.0:8000
```
启动成功之后，使用您的浏览器打开：
- **用户前台页面测试**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 
- **管理后台页面测试**: [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin) (使用上方创建的超级管理员账号进行登录)

---

## 📊 三、 数据分析与验证运行教程 (apple_music_research)

为了辅助验证本系统推荐算法的优越性，提供了独立的实验和结果可视化模块。

请另开一个终端窗口返回主目录并进入分析模块：
```bash
cd apple_music_research
pip install -r requirements.txt
```

### 本地科研图表生成
如果您需要复现或生成用于说明、评估算法推荐效果和数据流向的图表，直接执行：
```bash
python research_visualizer.py
```
这将会通过 Python （使用 matplotlib 等绘图工具）自动生成诸如：精准度对比 (Precision-Recall)、偏好统计分布以及跳过次数权重关联等分析图文件（如 `.png` 或 `.pdf`），可直接在论文中用以数据支撑。

---

## 💡 四、常见问题提示 (Q&A)

**Q1：C++ 依赖相关报错或 pip 无法安装怎么办？**  
请确保您系统中装载了基础编译工具，如遇类似 C++ 14 环境缺失的问题导致无法编译某些第三方包，您可以下载相关 wheel 配置或向开发者需求 C++ 支持包。

**Q2：打开推荐主页发现推荐内容不准确该怎么调试？**  
进入后台或前台利用 `createsuperuser` 或手动建立的账户多对库内歌曲点下喜爱或给出评分标签后，基于协同过滤(User-based 与 Item-based)的分析推荐列表将随之更新出现变化。
