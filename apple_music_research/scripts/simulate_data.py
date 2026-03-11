import os
import random
import django
from django.db import transaction

# 设置 Django 环境
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from user.models import User, Music, Tags, Rate

def setup_categories():
    """
    初始化歌曲分类标签（使用事务加速）
    """
    categories = ["流行", "摇滚", "古典", "爵士", "民谣"]
    tag_objs = []
    with transaction.atomic():
        for cat in categories:
            tag, _ = Tags.objects.get_or_create(name=cat)
            tag_objs.append(tag)
        
        all_musics = Music.objects.all()
        print(f"Assigning categories to {all_musics.count()} musics...")
        # 批量清空和添加标签在 Django M2M 中很难，这里精简操作
        for music in all_musics:
            # 只有没有标签时才打标签，避免重复操作耗时
            if not music.tags.exists():
                category = random.choice(tag_objs)
                music.tags.add(category)
    return list(Tags.objects.all())

def generate_clustered_data(num_users=2000, interactions_per_user=15):
    """
    使用批量创建加速生成具备兴趣簇特征的用户交互数据
    """
    tag_objs = setup_categories()
    
    print(f"Generating {num_users} users with clustering preference...")
    
    music_by_cat = {tag.id: list(Music.objects.filter(tags=tag)) for tag in tag_objs}
    all_music_ids = list(Music.objects.values_list('id', flat=True))
    
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    
    # 批量缓存用户和评分
    users_to_create = []
    rates_to_create = []
    
    with transaction.atomic():
        for i in range(num_users):
            uname = f"u_{''.join(random.choices(chars, k=6))}"
            user = User(
                username=uname, 
                name=uname,
                password="password123",
                phone="138" + "".join(random.choices("0123456789", k=8))
            )
            users_to_create.append(user)
            
        # 先批量创建用户以获得 ID
        created_users = User.objects.bulk_create(users_to_create)
        print(f"Successfully bulk created {len(created_users)} users.")

        for i, user in enumerate(created_users):
            # 为每个用户随机分配一个最爱类别
            favorite_cat = random.choice(tag_objs)
            target_musics = music_by_cat[favorite_cat.id]
            
            # 生成互动
            for _ in range(interactions_per_user):
                # 85% 概率选最爱类别，增加聚类效应！
                if random.random() < 0.85 and target_musics:
                    music = random.choice(target_musics)
                    is_fav = True
                else:
                    music_id = random.choice(all_music_ids)
                    music = Music.objects.get(id=music_id)
                    is_fav = music.tags.filter(id=favorite_cat.id).exists()

                total_duration = random.randint(180, 360)
                
                if is_fav:
                    # 强烈命中偏好！
                    mark = random.uniform(4.2, 5.0)
                    play_ratio = random.uniform(0.85, 1.0) # 听完
                    skip_count = 0
                else:
                    mark = random.uniform(1.0, 2.5)
                    play_ratio = random.uniform(0.05, 0.3) # 秒关
                    skip_count = random.randint(1, 3)
                
                play_duration = int(total_duration * play_ratio)
                
                rates_to_create.append(Rate(
                    user=user,
                    music=music,
                    mark=round(mark, 1),
                    play_duration=play_duration,
                    total_duration=total_duration,
                    skip_count=skip_count
                ))
            
            if i % 200 == 0:
                print(f"Prepared ratings for {i}/{num_users} users.")

        # 批量创建评分
        print(f"Bulk creating {len(rates_to_create)} ratings...")
        Rate.objects.bulk_create(rates_to_create)

if __name__ == "__main__":
    generate_clustered_data(num_users=1500, interactions_per_user=15)
    print("Optimization Complete: Clustered data injected!")
