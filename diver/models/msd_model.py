import numpy as np
import os
import pandas as pd
import plotly.offline as opy
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


N_JOBS = int(os.cpu_count() * 0.8)
FEATURE_COLUMNS = ["loudness", "tempo", "pitch_network_average_degree",
                   "pitch_network_entropy", "pitch_network_mean_clustering_coeff",
                   "timbre_00", "timbre_01", "timbre_02", "timbre_03", "timbre_04", "timbre_05", 
                   "timbre_06", "timbre_07", "timbre_08", "timbre_09", "timbre_10", "timbre_11"]


class MSDModel():
    def __init__(self,
                 df: pd.DataFrame,
                 feature_columns: list[str] = FEATURE_COLUMNS,
                 n_pca_components: int = 3,
                 n_neighbors_default: int = 10,
                 n_jobs: int = N_JOBS,
                 popularity_threshold: float = 0.85) -> None:
        """
        Initialize the class

        :param df: Dataframe of data we are fitting to
        :param features: List of features to use in the dataframe
        :param n_pca_components: Number of PCA components
        """
        # Capture the feature columns
        self.feature_columns = feature_columns

        # Copy of the dataframe
        self.orig_df = df[df["song_hotttnesss"] > 0].copy()
        self.mucked_with_df = self.orig_df.copy()

        # Remove NaN from artist familiarity
        self.mucked_with_df["artist_familiarity"] = self.mucked_with_df["artist_familiarity"].fillna(0)
        # Remove NaN from artist hotttnesss
        self.mucked_with_df["artist_hotttnesss"] = self.mucked_with_df["artist_hotttnesss"].fillna(0)

        # Standardize the data
        self.scalar = MaxAbsScaler()
        self.mucked_with_df[self.feature_columns] = self.scalar.fit_transform(self.mucked_with_df[self.feature_columns])

        # Fit the PCA
        self.pca = PCA(n_components=n_pca_components)
        self.pca.fit(self.mucked_with_df[self.feature_columns])

        # Filter out popular songs
        self.orig_df = self.orig_df[self.orig_df["song_hotttnesss"] < popularity_threshold]
        self.mucked_with_df = self.mucked_with_df[self.mucked_with_df["song_hotttnesss"] < popularity_threshold]
        self.pca_features = self.pca.transform(self.mucked_with_df[self.feature_columns])

        # Declare the nearest neighbors
        self.n_neighbors_default = n_neighbors_default
        self.nn = NearestNeighbors(n_neighbors=n_neighbors_default, n_jobs=n_jobs)
        self.nn.fit(self.pca_features)


    def find_k_neighbors(self,
                         user_df: pd.DataFrame,
                         n_neighbors: int = None) -> Tuple[pd.DataFrame, np.array]:
        """
        Given a user dataframe record, find the n closest neighbors

        :param user_df: The user data to fit to
        :param n_neighbors: Number of neighbors to return
        """
        # Make sure all columns we need exist
        assert all([x in user_df.columns for x in self.feature_columns])

        # Make sure only one row was passed in
        assert len(user_df) == 1

        # Copy the df
        df_in = user_df.copy()

        # Setting the number of neighbors the users want
        if not n_neighbors:
            n_neighbors = self.n_neighbors_default

        # Transform the user_df to our PCA
        df_in[self.feature_columns] = self.scalar.transform(df_in[self.feature_columns])
        transformed_user_array = self.pca.transform(df_in[self.feature_columns])

        # Run the model and return the results
        distances, idx = self.nn.kneighbors(transformed_user_array, n_neighbors=n_neighbors+1)
        
        # Test for edge case where one of the songs we are passing in
        # is a "dumpster", so retreive more and pair down after checking...
        if distances.flatten()[0] < 1e-6:
            distances = distances[1:]
            idx = idx.flatten()[1:]
        else:
            distances = distances[0:n_neighbors]
            idx = idx.flatten()[0:n_neighbors]
        return self.orig_df.iloc[idx.flatten()], transformed_user_array
    

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe with the features, add new columns with the features projected on
        each of the PCA axes

        :param df: DataFrame of info
        :return: New dataframe
        """
        # Perform the PCA
        df_out = df.copy()
        df_out[self.feature_columns] = self.scalar.transform(df_out[self.feature_columns])
        transformed_user_array = self.pca.transform(df_out[self.feature_columns])
        # Add the columns
        for j in range(transformed_user_array.shape[1]):
            df_out["PCA {}".format(j)] = transformed_user_array[:, j]

        return df_out


    def create_pca_plot(self,
                        user_dumpster_diver_features_df: pd.DataFrame,
                        msd_recs_dumpster_features_df: pd.DataFrame,
                        spotify_recs_dumpster_features_df: pd.DataFrame,
                        ) -> None:
        """
        Create the PCA plot with all the data
        :param user_dumpster_diver_features_df: User listening profile
        :param msd_recs_dumpster_features_df: MSD Dumpster Diver recos
        :param spotify_recs_dumpster_features_df: Spotify recos
        :returns:
        """
        # Transform all of the dfs to add the PCA columns
        background_pca_df = self.transform_df(self.orig_df)
        background_pca_df["Source"] = "MSD"

        user_dumpster_diver_features_df = self.transform_df(user_dumpster_diver_features_df)
        user_dumpster_diver_features_df["Source"] = "User"

        msd_recs_dumpster_features_df = self.transform_df(msd_recs_dumpster_features_df)
        msd_recs_dumpster_features_df["Source"] = "Dumpster Diver"

        spotify_recs_dumpster_features_df = self.transform_df(spotify_recs_dumpster_features_df)
        spotify_recs_dumpster_features_df["Source"] = "Spotify"

        df = pd.concat([background_pca_df,
                        user_dumpster_diver_features_df,
                        msd_recs_dumpster_features_df,
                        spotify_recs_dumpster_features_df])

        # Create the plots
        pca_plot = px.scatter_3d(
            df,
            x="PCA 0",
            y="PCA 1",
            z="PCA 2",
            template="plotly_dark",
            height=1200,
            color="song_hotttnesss",
            symbol="Source"
        )
        '''
        background_pca = px.scatter_3d(
            background_pca_df,
            x="PCA 0",
            y="PCA 1",
            z="PCA 2",
            template="plotly_dark",
            height=1200,
            color="song_hotttnesss"
        )
        background_pca.data[0].update(marker={"symbol": "circle-open"})

        user_pca = go.Scatter3d(
            user_dumpster_diver_features_df,
            x="PCA 0",
            y="PCA 1",
            z="PCA 2",
            template="plotly_dark",
            height=1200,
            opacity=0.5
        )
        user_pca.data[0].update(marker={"color": "yellow", "symbol": "diamond"})

        spotify_pca = px.scatter_3d(
            spotify_recs_dumpster_features_df,
            x="PCA 0",
            y="PCA 1",
            z="PCA 2",
            template="plotly_dark",
            height=1200
        )
        spotify_pca.data[0].update(marker={"color": "#1DB954", "symbol": "circle"})

        msd_pca = px.scatter_3d(
            msd_recs_dumpster_features_df,
            x="PCA 0",
            y="PCA 1",
            z="PCA 2",
            template="plotly_dark",
            height=1200
        )
        msd_pca.data[0].update(marker={"color": "#BC544B", "symbol": "square"})
        '''

        # Create the div
        div = opy.plot(pca_plot, auto_open=False, output_type="div")

        return div
