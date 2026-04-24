from django.urls import path
from . import views

urlpatterns = [
    path('', views.portal, name='portal'),
    path('dashboard/', views.index, name='index'),
    path('report/', views.report_view, name='report_view'),
    path('api/recommend/', views.recommend_api, name='recommend_api'),
    path('api/evaluate/', views.evaluate_api, name='evaluate_api'),
    path('api/report/', views.generate_report, name='generate_report'),
    path('search/', views.search_view, name='search'),
    path('music/<int:music_id>/', views.music_detail, name='music_detail'),
    path('profile/', views.profile_view, name='profile'),
    path('play/<int:music_id>/', views.play_api, name='play_api'),
    path('rate/<int:music_id>/', views.rate_api, name='rate_api'),
    path('login/', views.user_login, name='user_login'),
    path('register/', views.user_register, name='user_register'),
    path('logout/', views.user_logout, name='user_logout'),
]
