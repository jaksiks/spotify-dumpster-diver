from spotify.spotify_wrapper import SpotifyWrapper
from django.shortcuts import render
import os

# Create your views here.
def index(request):
    
    ## Call your data processing functions here
    ## This is what gets called when a user hits the website index/root directory "/"

    wrapper = SpotifyWrapper()
    tracks_df, pitch_network_df = wrapper.get_user_recent_tracks()

    # Extract seed artists, genres, and tracks
    seed_artists = tracks_df['artist_id'].tolist()
    seed_tracks = tracks_df['track_id'].tolist()

    flat_genres = list(set([genre for genres in tracks_df['genres'] for genre in genres]))
    seed_genres = flat_genres[:5]

    # Remove the 'song_array' column from the tracks DataFrame
    tracks_df_no_array = tracks_df.drop(columns=['song_array'])

    ## Then pass your processed data to the frontend via "context" below
    context = {
        ## Put data here that you want to pass to the frontend in key-value pair/dictionary form:
        ## 'key':variable,

        'tracks': tracks_df_no_array.to_html(),
        'pitch_network': pitch_network_df.to_html()

    }
    
    return render(request, 'dumpster_diver/index.html', context)
