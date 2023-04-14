from spotify.spotify_wrapper import SpotifyWrapper
from models.msd_model import MSDModel
from django.shortcuts import render
import os
import pandas as pd
import logging
import copy

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

    print(recommendations_df.columns)
    print(recommendations_df['song_id'])

    # Clean up recs for frontend
    clean_rec_df = copy.deepcopy(recommendations_df)
    rec_table_drop_columns = ['index', 'msd_id', 'artist_id', 'artist_familiarity', 'artist_hotttnesss', 'song_id', 'year', 'energy', 'danceability', 'tempo', 'pitch_network_average_degree', 'pitch_network_entropy','pitch_network_mean_clustering_coeff', 'timbre_00', 'timbre_01', 'timbre_02', 'timbre_03', 'timbre_04', 'timbre_05', 'timbre_06', 'timbre_07', 'timbre_08', 'timbre_09', 'timbre_10', 'timbre_11']
    clean_rec_df = clean_rec_df.drop(labels=rec_table_drop_columns, axis=1)
    rename_rec_columns_dict = {
        'artist_name':'Artist',
        'song_title':'Song Title',
        'song_hotttnesss':'Popularity',
        'loudness':'Loudness'
    }
    clean_rec_df.rename(columns=rename_rec_columns_dict, inplace=True)

    # Change column order
    clean_rec_df = clean_rec_df[['Song Title', 'Artist', 'Popularity', 'Loudness']]

    # Clean up Spotify top tracks for frontend
    clean_tracks_df = copy.deepcopy(tracks_df_no_array)
    tracks_table_drop_columns = ['track_id', 'artist_id', 'played_at', 'danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'genres', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature']
    clean_tracks_df = clean_tracks_df.drop(labels=tracks_table_drop_columns, axis=1)
    rename_tracks_columns_dict = {
        'name':'Song Title',
        'artist':'Artist',
        'popularity':'Popularity',
        'loudness':'Loudness'
    }
    clean_tracks_df.rename(columns=rename_tracks_columns_dict, inplace=True)

    # Normalize popularity column for comparison to hotttnesss
    clean_tracks_df['Popularity'] = clean_tracks_df['Popularity'] / 100

    ## Then pass your processed data to the frontend via "context" below
    context = {
        ## Put data here that you want to pass to the frontend in key-value pair/dictionary form:
        ## 'key':variable,
        'recommendations': clean_rec_df.to_html(classes='table table-bordered table-striped table-dark table-hover', table_id='rec-table', index=False),
        #'msd_plot': msd_plot,
        #'features': features,
        #'parallel_cords': parallel_cords,
        #'features_merged': features_merged,
        #'parallel_cords_merged': parallel_cords_merged,
        'tracks': clean_tracks_df.to_html(classes='table table-bordered table-striped table-dark table-hover', table_id='tracks-table', index=False),
        #'pitch_network': pitch_network_df.to_html()
    }
    
    return render(request, 'dumpster_diver/index.html', context)
