CREATE TABLE IF NOT EXISTS msd_table (
    msd_id TEXT PRIMARY KEY,
    artist_id TEXT NOT NULL,
    artist_name TEXT NOT NULL,
    artist_familiarity REAL NOT NULL,
    artist_hotttnesss REAL NOT NULL,
    song_id TEXT NOT NULL,
    song_title TEXT NOT NULL,
    song_hotttnesss TEXT NOT NULL,
    year INT NOT NULL,
    loudness REAL NOT NULL,
    energy REAL NOT NULL,
    danceability REAL NOT NULL,
    tempo REAL NOT NULL,
    pitch_network_average_degree REAL NOT NULL,
    pitch_network_entropy REAL NOT NULL,
    pitch_network_mean_clustering_coeff REAL NOT NULL,
    timbre_00 REAL NOT NULL,
    timbre_01 REAL NOT NULL,
    timbre_02 REAL NOT NULL,
    timbre_03 REAL NOT NULL,
    timbre_04 REAL NOT NULL,
    timbre_05 REAL NOT NULL,
    timbre_06 REAL NOT NULL,
    timbre_07 REAL NOT NULL,
    timbre_08 REAL NOT NULL,
    timbre_09 REAL NOT NULL,
    timbre_10 REAL NOT NULL,
    timbre_11 REAL NOT NULL
);

GRANT ALL PRIVILEGES ON DATABASE dumpster TO diver;
