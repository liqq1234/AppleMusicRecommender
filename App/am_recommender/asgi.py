"""
am_recommender 项目的 ASGI 配置。

它将 ASGI 可调用对象公开为一个名为 ``application`` 的模块级变量。

有关此文件的更多信息，请参阅：
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'am_recommender.settings')

application = get_asgi_application()
