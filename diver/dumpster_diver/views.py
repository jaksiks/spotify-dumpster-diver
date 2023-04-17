from spotify.spotify_wrapper import SpotifyWrapper
from models.msd_model import MSDModel
from django.shortcuts import render
import os
import pandas as pd
import logging
import numpy as np
import plotly.graph_objects as go
import pandas as pd
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

    
    pitchChart = generatePitchPlot(tracks_df, wrapper)

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
        'pitch_network': pitchChart
    }
    
    return render(request, 'dumpster_diver/index.html', context)


def generatePitchPlot(df, wrapper):
     #compile data for pitch network
    pitches = {
    0: "C",
    1: "C#/Db",
    2: "D",
    3: "D#/Eb",
    4: "E",
    5: "F",
    6: "F#/Gb",
    7: "G",
    8: "G#/Ab",
    9: "A",
    10: "A#/Bb",
    11: "B"}

    graph = 1
    pitch_df = df.copy()
    audio_analysis_cols = ["pitches", "timbres"]
    pitch_df[audio_analysis_cols] = pitch_df.apply(lambda x: wrapper.get_audio_analysis(x), axis=1)
    idx = 1
    print(pitch_df.iloc[idx])


    # plt.switch_backend("AGG")
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # plt.imshow(pitch_df.iloc[idx]["pitches"], aspect='auto')
    # plt.xlabel("Pitch")
    # plt.ylabel("Time")
    # plt.title(f"{pitch_df.iloc[idx]['name']} by {pitch_df.iloc[idx]['artist']}")
    # plt.xticks([x for x in pitches.keys()], pitches.values())
    # ax.set_xticks(np.arange(0, 13) - 0.5, minor=True)
    # plt.grid(which='minor', color='w', linestyle='--')
    
    # buffer = BytesIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)
    # image_png = buffer.getvalue()
    # graph = base64.b64encode(image_png)
    # graph = graph.decode('utf-8')
    # buffer.close()
    # return graph



# load dataset
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv")

# create figure
fig = go.Figure()

# Add surface trace
fig.add_trace(go.Surface(z=df.values.tolist(), colorscale="Viridis"))

# Update plot sizing
fig.update_layout(
    width=800,
    height=900,
    autosize=False,
    margin=dict(t=0, b=0, l=0, r=0),
    template="plotly_white",
)

# Update 3D scene options
fig.update_scenes(
    aspectratio=dict(x=1, y=1, z=0.7),
    aspectmode="manual"
)

# Add dropdown
fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=["type", "song1"],
                    label="Song1",
                    method="restyle"
                ),
                dict(
                    args=["type", "song2"],
                    label="Song2",
                    method="restyle"
                )
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)

# Add annotation
fig.update_layout(
    annotations=[
        dict(text="Trace type:", showarrow=False,
        x=0, y=1.085, yref="paper", align="left")
    ]
)

fig.show()
