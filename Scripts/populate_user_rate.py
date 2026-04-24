import os
import sys
import django
import random
from datetime import timedelta
from django.utils import timezone

# Set up Django environment
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'App'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'am_recommender.settings')
django.setup()

from user.models import User, Music, Rate, Tags

def populate():
    # 1. Create Tags
    tag_names = ['Pop', 'Rock', 'Jazz', 'Classical', 'Hip-Hop', 'Electronic', 'K-Pop', 'R&B']
    tags = [Tags.objects.get_or_create(name=t)[0] for t in tag_names]
    
    # 2. Create Musics (200 songs)
    print("Creating 200 songs...")
    musics = []
    for i in range(1, 201):
        m, created = Music.objects.get_or_create(
            name=f"Song_{i}",
            defaults={
                'artist': f"Artist_{random.randint(1, 20)}",
                'album': f"Album_{random.randint(1, 50)}",
                'years': f"{random.randint(2010, 2024)}",
                'bpm': random.randint(60, 180),
                'num': random.randint(0, 1000)
            }
        )
        if created:
            m.tags.add(random.choice(tags))
        musics.append(m)
        
    # 3. Create Users (50 users)
    print("Creating 50 users...")
    users = []
    for i in range(1, 51):
        u, created = User.objects.get_or_create(
            username=f"user{i}",
            defaults={
                'name': f"User_{i}",
                'password': 'password123'
            }
        )
        users.append(u)
        
    # 4. Create Interaction Records (~1800 records)
    print("Creating ~1800 interaction records...")
    Rate.objects.all().delete()
    count = 0
    now = timezone.now()
    
    for u in users:
        # User activity level (Long tail distribution)
        rand_val = random.random()
        if rand_val < 0.2: # High active
            record_count = random.randint(40, 60)
        elif rand_val < 0.8: # Average
            record_count = random.randint(15, 30)
        else: # Low active
            record_count = random.randint(5, 10)
            
        heard_songs = random.sample(musics, min(record_count, len(musics)))
        for m in heard_songs:
            play_ratio = random.uniform(0.1, 1.0)
            total_duration = random.randint(180, 400)
            play_duration = int(total_duration * play_ratio)
            
            # Explicit mark influenced by play ratio
            mark = 3.0
            if play_ratio > 0.8: mark = random.uniform(4.0, 5.0)
            elif play_ratio < 0.3: mark = random.uniform(1.0, 2.5)
            else: mark = random.uniform(2.5, 4.0)
            
            Rate.objects.create(
                user=u,
                music=m,
                mark=round(mark, 1),
                play_duration=play_duration,
                total_duration=total_duration,
                skip_count=1 if play_ratio < 0.3 else 0,
                create_time=now - timedelta(days=random.randint(0, 60))
            )
            count += 1
            
    print(f"Finished! Created {count} records.")

if __name__ == "__main__":
    populate()
