import pandas as pd
import yaml
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pathlib import Path


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
        scope = "user-read-recently-played user-library-read user-read-private user-read-email"

        # Authenticate with Spotify
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=self.client_id,
                                                            client_secret=self.client_secret,
                                                            redirect_uri=self.redirect_uri,
                                                            scope=scope))

    def get_user_recent_tracks(self, limit: int = 20) -> pd.DataFrame:
        """
        Retrieves the recently played songs by a user

        :param limit: The number of songs to return, DEFAULT of 20.
        :returns: DataFrame of the songs and their features
        """

        # Get the user's recently played songs
        recent_tracks = self.sp.current_user_recently_played(limit=limit)

        # Extract track IDs
        track_ids = [track['track']['id'] for track in recent_tracks['items']]

        # Get the track details including popularity
        track_details = [self.sp.track(track_id) for track_id in track_ids]

        # Get the audio features of the songs
        audio_features = self.sp.audio_features(track_ids)

        # ADDED: Get the song array data
        song_arrays = [self.sp.audio_analysis(track_id)['segments'] for track_id in track_ids]

        # TODO - May need to retrieve array data, not included in the audio_features

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
        for track, details, features, song_array, genres in zip(recent_tracks['items'], track_details, audio_features,
                                                                song_arrays, track_genres):
            track_info = {
                'track_id': track['track']['id'],
                'name': track['track']['name'],
                'artist': track['track']['artists'][0]['name'],
                'artist_id': track['track']['artists'][0]['id'],
                'played_at': track['played_at'],
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
