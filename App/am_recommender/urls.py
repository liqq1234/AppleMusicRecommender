"""
am_recommender 项目的 URL 配置。

`urlpatterns` 列表将 URL 路由到视图。有关更多信息，请参阅：
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
示例：
函数视图
    1. 添加导入： from my_app import views
    2. 向 urlpatterns 添加 URL： path('', views.home, name='home')
基于类的视图
    1. 添加导入： from other_app.views import Home
    2. 向 urlpatterns 添加 URL： path('', Home.as_view(), name='home')
包含另一个 URLconf
    1. 导入 include() 函数： from django.urls import include, path
    2. 向 urlpatterns 添加 URL： path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('user.urls')),
]
