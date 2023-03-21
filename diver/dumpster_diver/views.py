from spotify.spotify_wrapper import SpotifyWrapper
from django.shortcuts import render
import os

# Create your views here.
def index(request):
    
    ## Call your data processing functions here
    ## This is what gets called when a user hits the website index/root directory "/"

    wrapper = SpotifyWrapper()
    tracks_df = wrapper.get_user_recent_tracks()

    ## Then pass your processed data to the frontend via "context" below
    context = {
        ## Put data here that you want to pass to the frontend in key-value pair/dictionary form:
        ## 'key':variable,

        'tracks':tracks_df.to_html()
    }
    
    return render(request, 'dumpster_diver/index.html', context)
