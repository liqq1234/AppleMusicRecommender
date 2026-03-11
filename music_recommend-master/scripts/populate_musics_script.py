import random
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

django.setup()
from user.models import Music, Tags
import csv

publishers = ['Universal Music', 'Sony Music', 'Warner Music', 'Indie', 'Apple Music']

Music.objects.all().delete()
Tags.objects.all().delete()
opener = open('data/cloudmusic.csv', 'r', encoding='utf-8')
# opener.readline()
opener.readline()
reader = csv.reader(opener)
next(reader)
count = 0
for line in reader:
    if len(line) != 11:
        continue
    count += 1
    artist_id, artist_name, img_url, album_num, album_size, song_name, song_id, album_name, album_id, publish_time, lyric = line
    bpm = random.randint(60, 180)
    publisher = random.choice(publishers)
    Music.objects.get_or_create(name=song_name, defaults={'artist': artist_name, "pic": img_url, 'album': album_name, 'lyric': lyric, 'years': publish_time, 'bpm': bpm, 'publisher': publisher})
    print('success')
    if count > 10000:
        break
