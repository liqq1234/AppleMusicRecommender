import os
import sys
import django
import csv
import traceback
from django.utils import timezone
from django.db import connection

# Identify project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
APP_DIR = os.path.join(BASE_DIR, 'App')
DATA_DIR = os.path.join(BASE_DIR, 'Data')

# Set up Django environment
sys.path.append(APP_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'am_recommender.settings')
django.setup()

from user.models import User, Music, Rate

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

def import_music(metadata_path):
    if not os.path.exists(metadata_path):
        flush_print(f"Error: Music metadata file not found at {metadata_path}")
        return
    
    flush_print(f"Importing music from {metadata_path}...")
    Music.objects.all().delete()
    
    music_list = []
    with open(metadata_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                m = Music(
                    name=row['歌曲名称'][:128],
                    artist=row['艺术家'][:128],
                    album=row['专辑'][:128],
                    num=int(row['浏览量'] or 0),
                    years='2024',
                    bpm=0
                )
                music_list.append(m)
            except Exception:
                continue
            
            if len(music_list) >= 1000:
                Music.objects.bulk_create(music_list, ignore_conflicts=True)
                music_list = []
        
        if music_list:
            Music.objects.bulk_create(music_list, ignore_conflicts=True)
    
    flush_print(f"Finished Music import. Total: {Music.objects.count()}")

def import_interactions(interaction_path):
    if not os.path.exists(interaction_path):
        flush_print(f"Error: Interaction file not found at {interaction_path}")
        return

    flush_print(f"Importing interactions from {interaction_path}...")
    try:
        # Clear old interaction data and non-admin users
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM user_rate")
            cursor.execute("DELETE FROM user_user WHERE is_superuser = 0")
        
        # Pre-fetch Music map for matching
        flush_print("Pre-fetching tracks for mapping...")
        music_map = {m.name: m.id for m in Music.objects.all()}
        
        # Identify unique users to create in bulk
        flush_print("Scanning for unique users...")
        unique_usernames = set()
        with open(interaction_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unique_usernames.add(row['用户名'])
        
        flush_print(f"Creating {len(unique_usernames)} user profiles...")
        user_objects = [User(username=un, name=un, password='password123') for un in unique_usernames]
        User.objects.bulk_create(user_objects, ignore_conflicts=True)
        
        # Re-fetch user map
        user_map = {u.username: u.id for u in User.objects.all()}
        
        # Bulk import interactions
        rate_list = []
        now = timezone.now()
        match_count = 0
        
        flush_print("Processing 140k+ interactions (this may take a minute)...")
        with open(interaction_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                username = row['用户名']
                music_name = row['歌曲名称']
                
                user_id = user_map.get(username)
                music_id = music_map.get(music_name)
                
                if user_id and music_id:
                    match_count += 1
                    r = Rate(
                        user_id=user_id,
                        music_id=music_id,
                        mark=float(row.get('拟合兴趣评分') or 0),
                        play_duration=int(row.get('总播放时长(秒)') or 0),
                        total_duration=360,
                        create_time=now
                    )
                    rate_list.append(r)
                
                if len(rate_list) >= 2000:
                    Rate.objects.bulk_create(rate_list)
                    rate_list = []
                    if match_count % 20000 == 0:
                        flush_print(f"Imported {match_count} records...")
            
            if rate_list:
                Rate.objects.bulk_create(rate_list)

        flush_print(f"Finished Interactions. Matches: {match_count}, Final Count: {Rate.objects.count()}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    metadata_csv = os.path.join(DATA_DIR, 'music_library_metadata_106k.csv')
    interaction_csv = os.path.join(DATA_DIR, 'real_interaction_data_148k.csv')
    
    import_music(metadata_csv)
    import_interactions(interaction_csv)
