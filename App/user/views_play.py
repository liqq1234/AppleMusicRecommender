
def play_api(request, music_id):
    """记录播放交互 API"""
    from .models import Music, Rate, User
    from django.utils import timezone
    music = get_object_or_404(Music, id=music_id)
    
    # 获取当前用户ID (演示默认为 209)
    user_id = int(request.GET.get('user_id', 209))
    user = get_object_or_404(User, id=user_id)
    
    # 1. 增加歌曲播放量
    music.num += 1
    music.save()
    
    # 2. 记录或更新播放历史 (Rate)
    # 如果已经评价过，只更新时间；如果没评价过，创建一个带默认分的记录
    rate, created = Rate.objects.get_or_create(
        user=user, 
        music=music, 
        defaults={'mark': 1.0} # 设为 1.0 表示仅仅是“听过”，待用户评价
    )
    if not created:
        # 更新时间戳以进入“最近播放”
        rate.create_time = timezone.now()
        rate.save()
        
    return JsonResponse({
        'status': 'success', 
        'new_count': music.num,
        'message': f'Ready to play: {music.name}'
    })
