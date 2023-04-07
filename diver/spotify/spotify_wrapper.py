import pandas as pd
import yaml
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pathlib import Path
import plotly.express as px
import plotly.offline as opy
from sklearn.decomposition import PCA


class SpotifyWrapper:
    def __init__(self):
        # Read credentials from config.yml
        # spotify:
        #    id: "<id>"
        #    client_id: "<client_id>" 
        #    client_secret: "<client_secret>"
        #    redirect_uri: "http://localhost:8888/callback" <-- Make sure to add this to your Spotify Client!

        config_file = Path(__file__).parent / "config.yml"

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client_id = self.config["spotify"]["client_id"]
        self.client_secret = self.config["spotify"]["client_secret"]
        self.redirect_uri = self.config["spotify"]["redirect_uri"]

        # Set the required scopes for access
        scope = "user-read-recently-played user-library-read user-read-private user-read-email user-top-read"

        # Authenticate with Spotify
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=self.client_id,
                                                            client_secret=self.client_secret,
                                                            redirect_uri=self.redirect_uri,
                                                            scope=scope))

    def get_user_recent_tracks(self, limit: int = 5) -> pd.DataFrame:
        """
        Retrieves the recently played songs by a user

        :param limit: The number of songs to return, DEFAULT of 20.
        :returns: DataFrame of the songs and their features
        """

        # Get the user's recently played and top songs
        recent_tracks = self.sp.current_user_recently_played(limit=5)
        top_tracks = self.sp.current_user_top_tracks(limit=5)

        # Combine recent and top tracks
        combined_tracks = recent_tracks['items'] + [{'track': item} for item in top_tracks['items']]

        # Extract track IDs
        track_ids = [track['track']['id'] for track in combined_tracks]

        # Get the track details including popularity
        track_details = [self.sp.track(track_id) for track_id in track_ids]

        # Get the audio features of the songs
        audio_features = self.sp.audio_features(track_ids)

        # ADDED: Get the song array data
        song_arrays = [self.sp.audio_analysis(track_id)['segments'] for track_id in track_ids]

        # Get the genres of the songs
        track_genres = []
        for track_id in track_ids:
            track = self.sp.track(track_id)
            artist_id = track['artists'][0]['id']
            artist = self.sp.artist(artist_id)
            genres = artist.get('genres')
            track_genres.append(genres)

        # Combine track information and audio features
        combined_data = []
        for track, details, features, song_array, genres in zip(combined_tracks, track_details, audio_features,
                                                                song_arrays, track_genres):
            track_info = {
                'track_id': track['track']['id'],
                'name': track['track']['name'],
                'artist': track['track']['artists'][0]['name'],
                'artist_id': track['track']['artists'][0]['id'],
                'played_at': track.get('played_at', None),
                'popularity': details['popularity'],
                'song_array': song_array,
                'genres': genres
            }
            combined_data.append({**track_info, **features})

        # Store everything into a dataframe
        df = pd.DataFrame(combined_data)

        return df

    def get_spotify_recommendations(self,
                                    seed_artists=None, seed_genres=None, seed_tracks=None,
                                    limit=20, market=None,
                                    target_acousticness=None, target_duration_ms=None, target_instrumentalness=None,
                                    target_key=None, target_liveness=None, target_loudness=None, target_mode=None,
                                    target_popularity=None, target_speechiness=None, target_tempo=None,
                                    target_time_signature=None, target_valence=None, max_popularity=None
                                    ):

        params = {
            "limit": limit,
            "seed_artists": seed_artists if seed_artists else None,
            "seed_genres": seed_genres if seed_genres else None,
            "seed_tracks": seed_tracks if seed_tracks else None,
            "market": market,
            "target_acousticness": target_acousticness,
            "target_duration_ms": target_duration_ms,
            "target_instrumentalness": target_instrumentalness,
            "target_key": target_key,
            "target_liveness": target_liveness,
            "target_loudness": target_loudness,
            "target_mode": target_mode,
            "target_popularity": target_popularity,
            "target_speechiness": target_speechiness,
            "target_tempo": target_tempo,
            "target_time_signature": target_time_signature,
            "target_valence": target_valence,
            "max_popularity": max_popularity
        }

        # Remove None values from the params dictionary
        params = {k: v for k, v in params.items() if v is not None}

        recommendations = self.sp.recommendations(**params)

        if recommendations:
            track_ids = [track['id'] for track in recommendations['tracks']]

            # Get the track details including popularity
            track_details = [self.sp.track(track_id) for track_id in track_ids]

            # Get the audio features of the songs
            audio_features = self.sp.audio_features(track_ids)

            # Get the genres of the songs
            track_genres = []
            for track_id in track_ids:
                track = self.sp.track(track_id)
                artist_id = track['artists'][0]['id']
                artist = self.sp.artist(artist_id)
                genres = artist.get('genres')
                track_genres.append(genres)

            # Combine track information and audio features
            combined_data = []
            for track, details, features, genres in zip(recommendations['tracks'], track_details, audio_features,
                                                        track_genres):
                track_info = {
                    'track_id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'popularity': details['popularity'],
                    'genres': genres
                }
                combined_data.append({**track_info, **features})

            # Store everything into a dataframe
            df = pd.DataFrame(combined_data)
            return df

        else:
            # TODO: Add logging
            print("Error: Unable to fetch recommendations")
            return None

    def plot_song_data(self, df_songs, df_recs):
        if df_recs.empty:
            print("""""\nXXXXXXXXXXXXXXXX\n
            NO RECCOMENDATAIONS --- PASSING
            \nXXXXXXXXXXXXXXXX\n""")
            df_recs = df_songs
        print(f'Reccomended songs df backend = {df_recs.head()}')

        df_songs["tempo_normalized"] = df_songs["tempo"] / df_songs["tempo"].max()
        df_songs["popularity_normalized"] = df_songs["popularity"] / 100

        df_recs["tempo_normalized"] = df_recs["tempo"] / df_recs["tempo"].max()
        df_recs["popularity_normalized"] = df_recs["popularity"] / 100

        features = ["tempo_normalized", "popularity_normalized", "energy", "danceability","key"]

        fig = px.scatter_matrix(
            df_songs,
            dimensions=features,
            color="key",
            hover_name="name",
            template="plotly_dark",
            labels={"tempo_normalized": "Tempo",
                  "popularity_normalized": "Popularity Width", "energy": "Energy",
                  "danceability": "Danceability", "key": "Key", }
        )
        div = opy.plot(fig, auto_open=False, output_type='div')

        fig = px.parallel_coordinates(df_songs[features], 
                                    color="key",
                                    color_continuous_midpoint=3,
                                    template="plotly_dark",
                                    labels={"tempo_normalized": "Tempo",
                                            "popularity_normalized": "Popularity", "energy": "Energy",
                                            "danceability": "Danceability", "key": "Key", }
        )
        div2 = opy.plot(fig, auto_open=False, output_type='div')

        merged_df = pd.concat([df_songs, df_recs], axis=0, keys=['from_songs','from_recs']).reset_index(level=[0])
        features.append("level_0")
        print(f'merged_df {merged_df.head()}')
        fig = px.scatter_matrix(
            merged_df,
            dimensions=features,
            color="level_0",
            hover_name="name",
            template="plotly_dark",
            labels={"tempo_normalized": "Tempo",
                  "popularity_normalized": "Popularity Width", "energy": "Energy",
                  "danceability": "Danceability", "key": "Key", "level_0": "Source", }
        )
        div3 = opy.plot(fig, auto_open=False, output_type='div')

        fig = px.parallel_coordinates(merged_df[features], 
                                    color="key",
                                    color_continuous_midpoint=3,
                                    template="plotly_dark",
                                    labels={"tempo_normalized": "Tempo",
                                            "popularity_normalized": "Popularity", "energy": "Energy",
                                            "danceability": "Danceability", "key": "Key", "level_0": "Source",}
        )
        div4 = opy.plot(fig, auto_open=False, output_type='div')

        return div, div2, div3, div4
    
    def plot_msd(self):
        with open("spotify/msd.pkl", 'rb') as f:
            df = pd.read_pickle(f)

        # df = df[df['year'] >= 1990]
        # df = df[df['song_hotttnesss'] > 0]
        print(df)
        for col in df.columns:
            print(col)
        features = list(df.columns[8:])

        pca = PCA(n_components=3)
        components = pca.fit_transform(df[features])
        # fig = px.scatter_3d(
        #     components,
        #     x=0,
        #     y=1,
        #     z=2,
        #     color=df["year"],
        #     hover_name="song_title",
        #     template="plotly_dark"
        # )
        total_var = pca.explained_variance_ratio_.sum() * 100
        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=df['year'],
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
            hover_name=df["song_title"],
            template="plotly_dark",
            height=1200
        )       

        # fig = px.scatter_matrix(
        #     df.head(),
        #     dimensions=features,
        #     color="tempo",
        #     hover_name="song_title",
        #     template="plotly_dark"
        # )

        # fig = px.scatter_3d(df.sample(10000), x='pitch_network_average_degree', y='pitch_network_entropy', z='pitch_network_mean_clustering_coeff',
        #             color="tempo",
        #             template="plotly_dark",
        #             height=1200, 
        #             hover_name="song_title",
        #             animation_frame="year")
        # fig.update_layout(
        # scene = dict(
        #     xaxis = dict(range=[0,13],),
        #                 yaxis = dict(range=[0,1.75],),
        #                 zaxis = dict(range=[0,1],),)
        # )
        div = opy.plot(fig, auto_open=False, output_type='div')

        # fig = px.scatter_3d(df, x='pitch_network_average_degree', y='pitch_network_entropy', z='pitch_network_mean_clustering_coeff',
        #             color="tempo",
        #             template="plotly_dark",
        #             height=1200, 
        #             hover_name="song_title")
        # div = opy.plot(fig, auto_open=False, output_type='div')

        # fig = px.scatter_matrix(
        #     df.sample(1000),
        #     dimensions=features,
        #     color="year",
        #     template="plotly_dark",
        #     height=1200,
        #     hover_name="song_title"
        # )
        # df["year"] = df["year"].astype(str)
        # fig = px.parallel_coordinates(df[features].sample(1000), 
        #                             color="year",
        #                             template="plotly_dark",
        #                             height=1200
        #                             )

        # fig = px.scatter(df, x="pitch_network_average_degree", y="pitch_network_entropy", animation_frame="year",
        #                 color="tempo",
        #                 hover_name="song_title",
        #                 template="plotly_dark",
        #                 height=1200
        # )
        div = opy.plot(fig, auto_open=False, output_type='div')

        return div