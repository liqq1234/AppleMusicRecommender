"""
am_recommender 项目的 Django 配置。

由 'django-admin startproject' 使用 Django 5.2.12 生成（后降级为 2.2.1 以保证兼容性）。

有关此文件的更多信息，请参阅：
https://docs.djangoproject.com/en/5.2/topics/settings/

有关配置项及其取值的完整列表，请参阅：
https://docs.djangoproject.com/en/5.2/ref/settings/
"""

from pathlib import Path

# 在项目内部构建路径，例如：BASE_DIR / 'subdir'。
BASE_DIR = Path(__file__).resolve().parent.parent


# 快速启动开发设置 - 不适合生产环境
# 请参阅 https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# 安全警告：请确保在生产环境中使用的私钥安全！
SECRET_KEY = 'django-insecure-apple-music-recommendation-system-secret-key'

# 更改会话 Cookie 名称以避免与其他本地项目冲突
SESSION_COOKIE_NAME = 'am_sessionid'

# 安全警告：在生产环境中不要开启调试模式！
DEBUG = True

ALLOWED_HOSTS = []


# 应用程序定义

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'user',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'am_recommender.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'am_recommender.wsgi.application'


# 数据库配置
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': str(BASE_DIR / 'db.sqlite3'),
    }
}


# 密码验证
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# 国际化
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'zh-hans'

TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_TZ = True


# 静态文件 (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "static",
]

# 默认主键字段类型
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
