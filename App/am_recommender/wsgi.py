"""
am_recommender 项目的 WSGI 配置。

它将 WSGI 可调用对象公开为一个名为 ``application`` 的模块级变量。

有关此文件的更多信息，请参阅：
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'am_recommender.settings')

application = get_wsgi_application()
