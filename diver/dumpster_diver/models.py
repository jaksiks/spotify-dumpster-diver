from django.db import models

class SongList(models.Model):
    songs = models.CharField(max_length=20, verbose_name='Pitch Network')