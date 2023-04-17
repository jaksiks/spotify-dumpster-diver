import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as opy


new_columns = {'pitch_network_average_degree':'pitch_degree',
               'pitch_network_entropy': 'pitch_entropy',
               'pitch_network_mean_clustering_coeff': 'pitch_coeff',
               'song_hotttnesss': 'popularity'
               }


def generate_feature_plot(diver_reco_df: pd.DataFrame, spotify_reco_df: pd.DataFrame):
    features_to_plot = ['loudness', 'tempo', 'popularity', 'pitch_degree', 'pitch_entropy', 'pitch_coeff']
    row = 3
    col = 2

    diver_reco_df.rename(columns=new_columns, inplace=True)

    spotify_reco_df.rename(columns=new_columns, inplace=True)
    spotify_reco_df["popularity"] = spotify_reco_df["popularity"] / 100

    return subplot(diver_reco_df, spotify_reco_df, features_to_plot, row, col)


def subplot(final_MSD_recs, final_spo_recs, features, row, col):
    def histograms(final_MSD_recs, final_spo_recs, feature):
        # Create histogram for MSD dataframe
        trace1 = go.Histogram(
            x=final_MSD_recs[feature],
            nbinsx=25,  # specify number of bins
            name='MSD ' + feature,  # specify name for legend
            marker=dict(color='rgb(251, 128, 114)'),  # specify color for histogram bars
            opacity=0.7,  # specify opacity of histogram bars
            hovertext=final_MSD_recs["song_title"]
        )

        # Create histogram for Spotify dataframe
        trace2 = go.Histogram(
            x=final_spo_recs[feature],
            nbinsx=25,  # specify number of bins
            name='Spo '+ feature,  # specify name for legend
            marker=dict(color='rgb(179, 222, 105)'),  # specify color for histogram bars
            opacity=0.7,  # specify opacity of histogram bars
            hovertext=final_spo_recs["name"]
        )

        return trace1, trace2

    # Create subplots
    fig = make_subplots(rows=row, cols=col)

    # Add histograms as subplots
    for i, feature in enumerate(features):
        row_num = (i // col) + 1
        col_num = (i % col) + 1
        trace1, trace2 = histograms(final_MSD_recs, final_spo_recs, feature)
        fig.add_trace(trace1, row=row_num, col=col_num)
        fig.add_trace(trace2, row=row_num, col=col_num)

        fig.update_xaxes(title_text=feature, row=row_num, col=col_num)
        fig.update_yaxes(title_text='Distribution', row=row_num, col=col_num)

    fig.update_layout(
        height=600,
        title_text="MSD and Spotify Feature Data",
        title_x=0.5,
        template="plotly_dark"
    )
    
    div = opy.plot(fig, auto_open=False, output_type="div")

    return div
