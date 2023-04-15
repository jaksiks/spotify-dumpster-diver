from spotify.spotify_wrapper import SpotifyWrapper
from models.msd_model import MSDModel
from django.shortcuts import render
import os
import pandas as pd
import logging
import copy

# Create your views here.
def index(request):
    # Set up logging
    logger = logging.getLogger()
        
    # Initialize the SpotifyWrapper
    logger.info("Retrieving User Spotify Songs")
    wrapper = SpotifyWrapper()
    
    # Fetch user's recent tracks and features
    tracks_df, dumpster_diver_feature_df = wrapper.get_user_recent_tracks(top_tracks_limit=5)

    # Define parameters for Spotify recommendations
    seed_genres = list(set([genre for genres in tracks_df['genres'] for genre in genres]))[:5]
    sample_params = {
        "limit": 15,
        "seed_genres": seed_genres,
    }

    # Get Spotify recommendations and clean up the dataframe
    spotify_recs_df = wrapper.get_spotify_recommendations(**sample_params)
    rename_columns_dict = {
        'name': 'Song Title',
        'artist': 'Artist',
        'popularity': 'Popularity',
        'loudness': 'Loudness'
    }
    cleaned_spotify_recs_df = clean_dataframe(spotify_recs_df, rename_columns=rename_columns_dict)

    #TODO: Add the spotify recs to the PCA plot

    # Load the MSDModel
    df_filepath = "spotify/msd.pkl"
    logger.info("Loading the MSD Model: {}".format(df_filepath))
    df = pd.read_pickle(df_filepath)
    msd_model = MSDModel(df)

    # Get the MSD recommendations and transform data
    logger.info("Diving into the dumpster!")
    recommendations_list_dfs = []
    transformed_song_list_dfs = []
    for i in range(len(dumpster_diver_feature_df)):
        cur_rec_df, transformed_song_df = msd_model.find_k_neighbors(
            dumpster_diver_feature_df.iloc[i:i+1],
            n_neighbors=3
        )
        recommendations_list_dfs.append(cur_rec_df)
        transformed_song_list_dfs.append(transformed_song_df)

    # Clean up the MSD recommendations and user's recent tracks data
    recommendations_df = pd.concat(recommendations_list_dfs).reset_index()
    rename_columns_dict = {
        'artist_name': 'Artist',
        'song_title': 'Song Title',
        'song_hotttnesss': 'Popularity',
        'loudness': 'Loudness'
    }
    clean_rec_df = clean_dataframe(recommendations_df, rename_columns=rename_columns_dict)
    rename_columns_dict = {
        'name': 'Song Title',
        'artist': 'Artist',
        'popularity': 'Popularity',
        'loudness': 'Loudness'
    }
    clean_tracks_df = clean_dataframe(tracks_df.drop(columns=['song_array']), tracks=True,
                                      rename_columns=rename_columns_dict)

    # TODO: Plots and plots and plots
    # logger.info("Displaying our Dumpster Finds!")
    # msd_plot = wrapper.plot_msd()
    # features, parallel_cords, features_merged, parallel_cords_merged  = wrapper.plot_song_data(tracks_df, recommendations_df)

    # Prepare the data to be passed to the frontend
    context = {
        ## Put data here that you want to pass to the frontend in key-value pair/dictionary form:
        ## 'key':variable,
        'recommendations': clean_rec_df.to_html(classes='table table-bordered table-striped table-dark table-hover', table_id='rec-table', index=False),
        'spotify_recs': cleaned_spotify_recs_df.to_html(classes='table table-bordered table-striped table-dark table-hover', table_id='rec-table', index=False),
        'tracks': clean_tracks_df.to_html(classes='table table-bordered table-striped table-dark table-hover', table_id='tracks-table', index=False),
        # 'msd_plot': msd_plot,
        # 'features': features,
        # 'parallel_cords': parallel_cords,
        # 'features_merged': features_merged,
        # 'parallel_cords_merged': parallel_cords_merged,
        # 'pitch_network': pitch_network_df.to_html()
    }

    return render(request, 'dumpster_diver/index.html', context)

def clean_dataframe(df, tracks=False, rename_columns=None):
    drop_columns = ['index', 'msd_id', 'artist_id', 'artist_familiarity', 'artist_hotttnesss', 'song_id',
                    'year', 'energy', 'danceability', 'tempo', 'pitch_network_average_degree',
                    'pitch_network_entropy', 'pitch_network_mean_clustering_coeff', 'timbre_00', 'timbre_01',
                    'timbre_02', 'timbre_03', 'timbre_04', 'timbre_05', 'timbre_06', 'timbre_07', 'timbre_08',
                    'timbre_09', 'timbre_10', 'timbre_11']

    cleaned_df = df.drop(labels=[col for col in drop_columns if col in df.columns], axis=1)

    if rename_columns:
        cleaned_df.rename(columns=rename_columns, inplace=True)

    if tracks:
        cleaned_df['Popularity'] = cleaned_df['Popularity'] / 100

    cleaned_df = cleaned_df[['Song Title', 'Artist', 'Popularity', 'Loudness']]
    return cleaned_df
