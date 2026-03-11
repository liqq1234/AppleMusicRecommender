import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from user.models import Rate, User, Music
import pandas as pd

def check_stats():
    total_rates = Rate.objects.count()
    users = User.objects.count()
    musics = Music.objects.count()
    
    print(f"Total Ratings: {total_rates}")
    print(f"Total Users: {users}")
    print(f"Total Musics: {musics}")
    
    if users > 0 and musics > 0:
        sparsity = 1 - (total_rates / (users * musics))
        print(f"Data Sparsity: {sparsity*100:.2f}%")

    # 检查评分分布
    rates = list(Rate.objects.values_list('mark', flat=True))
    if rates:
        print("\nRating Distribution:")
        print(pd.Series(rates).value_counts().sort_index())

if __name__ == "__main__":
    check_stats()
