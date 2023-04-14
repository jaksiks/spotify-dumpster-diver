from spotify.spotify_wrapper import SpotifyWrapper
from models.msd_model import MSDModel
from django.shortcuts import render
import os
import pandas as pd
import logging

# Create your views here.
def index(request):
    logger = logging.getLogger()
        
    ## Call your data processing functions here
    ## This is what gets called when a user hits the website index/root directory "/"
    logger.info("Retrieving User Spotify Songs")
    wrapper = SpotifyWrapper()
    
    tracks_df, dumpster_diver_feature_df = wrapper.get_user_recent_tracks(top_tracks_limit=5)

    # Extract seed artists, genres, and tracks
    seed_artists = tracks_df['artist_id'].tolist()
    seed_tracks = tracks_df['track_id'].tolist()

    flat_genres = list(set([genre for genres in tracks_df['genres'] for genre in genres]))
    seed_genres = flat_genres[:5]

    # TODO: DELETE!
    sample_params = {
        "limit": 10,
        "seed_artists": None,
        "seed_genres": seed_genres,
        "seed_tracks": None,
        "market": None,
        "target_acousticness": 0.8,
        "target_duration_ms": 200000,
        "target_instrumentalness": 0.5,
        "target_key": 5,
        "target_liveness": 0.3,
        "target_loudness": -15,
        "target_mode": 1,
        "target_popularity": 0,
        "target_speechiness": 0.1,
        "target_tempo": 120,
        "target_time_signature": 4,
        "target_valence": 0.5,
        "max_popularity": 50
    }

    # Load the model
    df_filepath = "spotify/msd.pkl"
    logging.info("Loading our Dataframe: {}".format(df_filepath))
    df = pd.read_pickle(df_filepath)
    logging.info("Creating the Dumpster Diver Model")
    msd_model = MSDModel(df)

    # // TODO: Replace the sample song recommendations with what is returned by MSD
    # // This will go into the wrapper.plot_song_data function below

    #TODO: Add the spotify recos to the PCA plot
    #recommendations_df = wrapper.get_spotify_recommendations(**sample_params)
    # Pass the seed artists, genres, tracks, and targets into the recommendations function
    logging.info("Diving into the dumpster!")
    recommendations_list_dfs = []
    transformed_song_list_dfs = []
    for i in range(len(dumpster_diver_feature_df)):
        cur_rec_df, transformed_song_df = msd_model.find_k_neighbors(
            dumpster_diver_feature_df.iloc[i:i+1],
            n_neighbors=3
        )        
        recommendations_list_dfs.append(cur_rec_df)
        transformed_song_list_dfs.append(transformed_song_df)
    recommendations_df = pd.concat(recommendations_list_dfs).reset_index()

    logger.info("Displaying our Dumpster Finds!")
    #msd_plot = wrapper.plot_msd()
    #features, parallel_cords, features_merged, parallel_cords_merged  = wrapper.plot_song_data(tracks_df, recommendations_df)
    
    # Remove the 'song_array' column from the tracks DataFrame
    tracks_df_no_array = tracks_df.drop(columns=['song_array'])

    ## Then pass your processed data to the frontend via "context" below
    context = {
        ## Put data here that you want to pass to the frontend in key-value pair/dictionary form:
        ## 'key':variable,
        'recommendations': recommendations_df.to_html(),
        #'msd_plot': msd_plot,
        #'features': features,
        #'parallel_cords': parallel_cords,
        #'features_merged': features_merged,
        #'parallel_cords_merged': parallel_cords_merged,
        'tracks': tracks_df_no_array.to_html(),
        #'pitch_network': pitch_network_df.to_html()
    }
    
    return render(request, 'dumpster_diver/index.html', context)
