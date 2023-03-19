import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Read credentials from config.txt
# config.txt format:
# SPOTIPY_CLIENT_ID=
# SPOTIPY_CLIENT_SECRET=
# SPOTIPY_REDIRECT_URI=
config = {}
with open('config.txt', 'r') as file:
    for line in file:
        if line.strip() == "":
            continue
        key, value = line.strip().split("=")
        config[key] = value.strip()

client_id = config['SPOTIPY_CLIENT_ID']
client_secret = config['SPOTIPY_CLIENT_SECRET']
redirect_uri = config['SPOTIPY_REDIRECT_URI']

# Set the required scopes for access
# Note: it would not work with just user-read-recently-played
scope = "user-read-recently-played user-library-read"

# Authenticate with Spotify
# Adjust request_timeout if needed
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope), requests_timeout=140)

# Get the user's 20 recently played songs
# Adjust number if needed
recent_tracks = sp.current_user_recently_played(limit=20)

# Extract track IDs
track_ids = [track['track']['id'] for track in recent_tracks['items']]

# Get the audio features of the songs
audio_features = sp.audio_features(track_ids)

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

# Write the dataframe to an Excel file
file_name = "recently_played_songs.xlsx"
df.to_excel(file_name, index=False)

print(f"Dataframe successfully written to {file_name}")

