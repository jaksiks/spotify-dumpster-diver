import numpy as np
import os
import pandas as pd
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
                 n_jobs: int = N_JOBS) -> None:
        """
        Initialize the class

        :param df: Dataframe of data we are fitting to
        :param features: List of features to use in the dataframe
        :param n_pca_components: Number of PCA components
        """
        # Capture the feature columns
        self.feature_columns = feature_columns

        # Copy of the dataframe
        self.orig_df = df.copy()
        self.mucked_with_df = df.copy()

        # Standardize the data
        self.scalar = MaxAbsScaler()
        self.mucked_with_df[self.feature_columns] = self.scalar.fit_transform(self.mucked_with_df[self.feature_columns])

        # Fit the PCA
        self.pca = PCA(n_components=n_pca_components)
        self.pca_features = self.pca.fit_transform(self.mucked_with_df[self.feature_columns])

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
        transformed_user_df = self.pca.transform(df_in[self.feature_columns])

        # Run the model and return the results
        distances, idx = self.nn.kneighbors(transformed_user_df, n_neighbors=n_neighbors)
        return self.orig_df.iloc[idx.flatten()], transformed_user_df
