import os
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


N_JOBS = int(os.cpu_count() * 0.8)
FEATURE_COLUMNS = ["artist_familiarity", "artist_hotttnesss", "loudness", "tempo",
                   "pitch_network_average_degree",
                   "pitch_network_entropy", "pitch_network_mean_clustering_coeff"]


class MSDModel():
    def __init__(self,
                 df: pd.DataFrame,
                 feature_columns: list[str] = FEATURE_COLUMNS,
                 n_pca_components: int = 3,
                 n_neighbors_default: int = 10,
                 n_jobs: int = N_JOBS) -> None:
        """
        :param df: Dataframe of data we are fitting to
        :param features: List of features to use in the dataframe
        :param n_pca_components: Number of PCA components
        """
        # Capture the feature columns
        self.feature_columns = feature_columns

        # Copy of the dataframe
        self.df = df.copy()

        # Standardize the data
        self.scalar = MaxAbsScaler()
        self.df[self.feature_columns] = self.scalar.fit_transform(self.df[self.feature_columns])

        # Fit the PCA
        self.pca = PCA(n_pca_components=n_pca_components)
        self.pca_features = self.pca.fit_transform(self.df[self.feature_columns])

        # Declare the nearest neighbors
        self.nn = NearestNeighbors(n_neighbors=n_jobs)

    def find_k_neighbors(self,
                         user_df: pd.DataFrame,
                         n_neighbors: int = None):
        assert all([x in user_df.columns for x in self.feature_columns])
        if not n_neighbors:
            n_neighbors = self.n_neighbors_default

        self.nn.kneighbors()
    
    