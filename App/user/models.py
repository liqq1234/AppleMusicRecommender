from django.db import models

class User(models.Model):
    username = models.CharField(max_length=128, unique=True)
    password = models.CharField(max_length=128)
    name = models.CharField(max_length=128, unique=True)
    email = models.EmailField(max_length=254, blank=True, null=True)

    def __str__(self):
        return self.name

class Tags(models.Model):
    name = models.CharField(max_length=128)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

class Music(models.Model):
    name = models.CharField(max_length=128, unique=True)
    artist = models.CharField(max_length=128)
    album = models.CharField(max_length=128)
    years = models.CharField(max_length=128)
    bpm = models.IntegerField(default=0)
    num = models.IntegerField(default=0, verbose_name="浏览量")
    sump = models.IntegerField(default=0, verbose_name="收藏人数")
    publisher = models.CharField(max_length=128, blank=True, null=True)
    tags = models.ManyToManyField(Tags, blank=True)
    collect = models.ManyToManyField(User, related_name="collected_musics", blank=True)

    def __str__(self):
        return self.name

class Rate(models.Model):
    music = models.ForeignKey(Music, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    mark = models.FloatField(verbose_name="评分")
    play_duration = models.IntegerField(verbose_name="播放时长(秒)", default=0)
    total_duration = models.IntegerField(verbose_name="歌曲总时长(秒)", default=1)
    skip_count = models.IntegerField(verbose_name="跳过次数", default=0)
    create_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.name} - {self.music.name} ({self.mark})"

class Comment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    music = models.ForeignKey(Music, on_delete=models.CASCADE)
    content = models.TextField()
    create_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.name} on {self.music.name}"
