import pandas as pd
import yaml
import spotipy
from spotipy.oauth2 import SpotifyOAuth


class SpotifyWrapper:
    def __init__(self, config_file: str = "config.yml"):
        # Read credentials from config.yml
        #spotify:
        #    id: "<id>"
        #    client_id: "<client_id>" 
        #    client_secret: "<client_secret>"
        #    redirect_uri: "http://localhost:8888/callback" <-- Make sure to add this to your Spotify Client!
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client_id = self.config["spotify"]["client_id"]
        self.client_secret = self.config["spotify"]["client_secret"]
        self.redirect_uri = self.config["spotify"]["redirect_uri"]

    def get_user_recent_tracks(self, limit: int = 20) -> pd.DataFrame:
        """
        Retrieves the recently played songs by a user

        :param limit: The number of songs to return, DEFAULT of 20.
        :returns: DataFrame of the songs and their features
        """
        # Set the required scopes for access
        # Note: it would not work with just user-read-recently-played
        scope = "user-read-recently-played user-library-read"

        # Authenticate with Spotify
        # Adjust request_timeout if needed
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=self.client_id,
                                                       client_secret=self.client_secret,
                                                       redirect_uri=self.redirect_uri,
                                                       scope=scope),
                             requests_timeout=140)

        # Get the user's recently played songs
        recent_tracks = sp.current_user_recently_played(limit=limit)

        # Extract track IDs
        track_ids = [track['track']['id'] for track in recent_tracks['items']]

        # Get the audio features of the songs
        audio_features = sp.audio_features(track_ids)

        # TODO - May need to retrieve array data, not included in the audio_features

        # Combine track information and audio features
        combined_data = []
        for track, features in zip(recent_tracks['items'], audio_features):
            track_info = {
                'track_id': track['track']['id'],
                'name': track['track']['name'],
                'artist': track['track']['artists'][0]['name'],
                'played_at': track['played_at']  # maybe this will be useful for dataviz
            }
            combined_data.append({**track_info, **features})

        # Store everything into a dataframe
        df = pd.DataFrame(combined_data)
        
        return df
