from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.db import models
from .recommend_engine import recommend_for_user_svd
from .evaluator import evaluate_all_models
from .research_visualizer import generate_academic_charts
import os

def index(request):
    """核心推荐引擎仪表盘"""
    # 逻辑：优先从 URL 获取 UID (用于测试切换)，否则使用 Session，否则使用默认值 209
    user_id = request.GET.get('user_id')
    if user_id:
        user_id = int(user_id)
        # 如果是手动切换用户，同步更新到 Session 中
        request.session['user_id'] = user_id
    else:
        user_id = request.session.get('user_id', 209)

    top_n = int(request.GET.get('top_n', 10))
    
    # 获取 SVD 推荐结果
    recommendations = recommend_for_user_svd(user_id, top_n)
    
    # 丰富推荐理由 (演示用)
    reasons = ["基于您的历史偏好相似度", "近期热门单曲推荐", "同一流派艺术家发现", "协同过滤预测结果"]
    for i, m in enumerate(recommendations):
        m.reason = reasons[i % len(reasons)]
        
    from .models import Rate, User, Music
    user = get_object_or_404(User, id=user_id)
    history = Rate.objects.filter(user=user).select_related('music').order_by('-create_time')[:10]
    
    stats = {
        'total_music': Music.objects.count(),
        'total_users': User.objects.count(),
        'total_interactions': Rate.objects.count()
    }
    
    # 获取活跃测试用户用于下拉框
    from django.db.models import Count
    active_users = User.objects.annotate(rate_count=Count('rate')).filter(rate_count__gt=5).order_by('-rate_count')[:50]
    
    context = {
        'recommendations': recommendations,
        'history': history,
        'user': user,
        'stats': stats,
        'selected_uid': user_id,
        'user_id': user_id,  # 修复：供用户 ID 输入框使用
        'active_users': active_users,
        'top_n': top_n,
        'is_authenticated': 'user_id' in request.session,
        'user_name': user.name
    }
    return render(request, 'user/index.html', context)

def user_login(request):
    """登录逻辑"""
    error = None
    if request.method == 'POST':
        uname = request.POST.get('username')
        pswd = request.POST.get('password')
        from .models import User
        user = User.objects.filter(username=uname, password=pswd).first()
        if user:
            request.session['user_id'] = user.id
            request.session['user_name'] = user.name
            return redirect('index')
        else:
            error = "用户名或密码错误"
    return render(request, 'user/login.html', {'error': error})

def user_register(request):
    """注册逻辑"""
    error = None
    if request.method == 'POST':
        uname = request.POST.get('username')
        name = request.POST.get('name')
        email = request.POST.get('email')
        pswd = request.POST.get('password')
        re_pswd = request.POST.get('re_password')
        
        from .models import User
        if User.objects.filter(username=uname).exists():
            error = "用户名已存在"
        elif pswd != re_pswd:
            error = "两次输入的密码不一致"
        else:
            User.objects.create(username=uname, name=name, email=email, password=pswd)
            return redirect('user_login')
            
    return render(request, 'user/register.html', {'error': error})

from django.http import JsonResponse

def rate_api(request, music_id):
    """评分接口"""
    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'status': 'error', 'code': 'unauth', 'message': 'Please login first'})
    
    mark = float(request.GET.get('mark', 5.0))
    user = User.objects.get(id=user_id)
    music = Music.objects.get(id=music_id)
    
    rate, created = Rate.objects.update_or_create(
        user=user,
        music=music,
        defaults={'mark': mark}
    )
    
    return JsonResponse({'status': 'success', 'new_mark': mark})

def user_logout(request):
    """注销逻辑"""
    request.session.flush()
    return redirect('portal')

def portal(request):
    """项目门户页面"""
    context = {
        'is_authenticated': 'user_id' in request.session,
        'user_name': request.session.get('user_name')
    }
    return render(request, 'user/portal.html', context)

def report_view(request):
    """学术评估报告页面"""
    return render(request, 'user/report.html')

def recommend_api(request):
    user_id = request.session.get('user_id', 209)
    recs = recommend_for_user_svd(user_id)
    data = [{'id': m.id, 'name': m.name, 'artist': m.artist} for m in recs]
    return JsonResponse({'status': 'success', 'data': data})

def evaluate_api(request):
    metrics = evaluate_all_models()
    return JsonResponse({'status': 'success', 'metrics': metrics})

def generate_report(request):
    try:
        chart_paths = generate_academic_charts()
        return JsonResponse({'status': 'success', 'charts': chart_paths})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

def search_view(request):
    """全局搜索"""
    query = request.GET.get('q', '')
    from .models import Music
    if query:
        results = Music.objects.filter(
            models.Q(name__icontains=query) | 
            models.Q(artist__icontains=query) |
            models.Q(album__icontains=query)
        )[:20]
    else:
        results = []
    # 增加登录状态
    return render(request, 'user/search.html', {
        'results': results, 
        'query': query,
        'user_name': request.session.get('user_name'),
        'is_authenticated': 'user_id' in request.session
    })

def music_detail(request, music_id):
    """音乐详情页"""
    from .models import Music, Rate, Comment
    music = get_object_or_404(Music, id=music_id)
    
    # 获取相关推荐 (同歌手)
    related = Music.objects.filter(artist=music.artist).exclude(id=music_id)[:6]
    
    # 模拟数据
    comments = Comment.objects.filter(music=music).order_by('-create_time')
    plays = music.num
    avg_score = Rate.objects.filter(music=music).aggregate(models.Avg('mark'))['mark__avg'] or 0.0
    
    context = {
        'music': music,
        'related': related,
        'comments': comments,
        'plays': plays,
        'avg_score': round(avg_score, 1),
        'user_name': request.session.get('user_name'),
        'is_authenticated': 'user_id' in request.session
    }
    return render(request, 'user/detail.html', context)

def profile_view(request):
    """用户个人中心"""
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('user_login')
        
    from .models import User, Rate, Music
    user = get_object_or_404(User, id=user_id)
    
    # 交互记录
    history = Rate.objects.filter(user=user).select_related('music').order_by('-create_time')[:20]
    favorites = Rate.objects.filter(user=user, mark__gte=4.0).select_related('music').order_by('-mark')[:10]
    
    # 为你推荐 (直接调用引擎)
    recommendations = recommend_for_user_svd(user_id, top_n=10)
    
    stats = {
        'total_listened': Rate.objects.filter(user=user).count(),
        'top_genre': "Pop / Jazz" # 模拟逻辑
    }
    
    context = {
        'view_user': user,
        'history': history,
        'favorites': favorites,
        'recommendations': recommendations,
        'stats': stats,
        'is_authenticated': True
    }
    return render(request, 'user/profile.html', context)

def play_api(request, music_id):
    """记录播放交互 API"""
    from .models import Music, Rate, User
    from django.utils import timezone
    music = get_object_or_404(Music, id=music_id)
    
    user_id = request.session.get('user_id', 209)
    user = get_object_or_404(User, id=user_id)
    
    music.num += 1
    music.save()
    
    rate, created = Rate.objects.get_or_create(
        user=user, 
        music=music, 
        defaults={'mark': 1.0}
    )
    if not created:
        rate.create_time = timezone.now()
        rate.save()
        
    return JsonResponse({
        'status': 'success', 
        'new_count': music.num,
        'message': f'Ready to play: {music.name}'
    })
