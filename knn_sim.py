#--------- Imports  ---------

import spotipy
import warnings
import pandas as pd
import streamlit as st
import plotly.express as px 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from spotipy.oauth2 import SpotifyClientCredentials

warnings.filterwarnings("ignore")

#--------- Constants  ---------
st.title("Spotify Playlist Recommender")

CLIENT_ID = "3e28cbdec0e841c08ec2d6b9b949cef1"
CLIENT_SECRET = "857ea858725348e09cbeb6f1470bcbac"

COLS = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'valence']


POP1 = '37i9dQZF1DXcBWIGoYBM5M'
HH1 = '37i9dQZF1DX0XUsuxWHRQd'
COUNTRY1 = '37i9dQZF1DX1lVhptIYRda'
LATIN1 = '37i9dQZF1DX10zKzsJ2jva'
ROCK1 = '37i9dQZF1DXcF6B6QPhFDv'
EDM1 = '37i9dQZF1DX4dyzvuaRJ0n'
INDIE1 = '37i9dQZF1DXdwmD5Q7Gxah'
RB1 = '37i9dQZF1DX4SBhb3fqCJd'
JAZZ1 = '37i9dQZF1DX7YCknf2jT6s'
METAL1 = '37i9dQZF1DWXNFSTtym834'
playlist_arr = [POP1, HH1, COUNTRY1, LATIN1, ROCK1, EDM1, INDIE1, RB1, JAZZ1, METAL1]
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



#--------- Functions  ---------

def retrieve_spotify_data(playlist_uri_array): 
    data = []

    for playlist_uri in playlist_uri_array:
        playlist_id = playlist_uri.split(':')[-1]
        playlist = sp.playlist(playlist_id)
        tracks = playlist['tracks']['items']

        for track in tracks:
            track_info = track['track']
            track_name = track_info['name']
            artist_name = track_info['artists'][0]['name']
            audio_features = sp.audio_features(track_info['id'])[0]
            genres = sp.artist(track_info['artists'][0]['id'])['genres']

            if audio_features is not None and genres is not None:
                audio_data = {
                    'track_name': track_name,
                    'artist_name': artist_name,
                    'genres': genres
                }
                audio_data.update(audio_features)
                data.append(audio_data)
    playlist_df = pd.DataFrame(data)
    playlist_df = playlist_df.drop(['key', 'mode', 'liveness', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'], axis=1)
    playlist_df.to_csv('data/music_data.csv', index=False)
    return playlist_df

#playlist_df = retrieve_spotify_data(playlist_arr)
playlist_df = pd.read_csv('data/music_data.csv')
st.dataframe(playlist_df)

def X_y_split(playlist_df):
    X = playlist_df[COLS]
    y = playlist_df['genres']
    return X,y

X, y = X_y_split(playlist_df)

def silhouette_graph(X):
    range_n_clusters = [5, 10, 15, 20, 25, 30, 35, 40]
    silhouette_avg = []
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
    
        silhouette_avg.append(silhouette_score(X, cluster_labels))

    fig, ax = plt.subplots()
    ax.plot(range_n_clusters, silhouette_avg, 'bx-')
    ax.set_xlabel('Values of K')
    ax.set_ylabel('Silhouette score')
    ax.set_title('Silhouette analysis For Optimal k')
    return fig

# Assuming you have the data 'X' available

# Create a Streamlit app
st.title('Silhouette Analysis')
st.write('This app calculates and displays the silhouette scores for different values of K in KMeans clustering.')

# Generate the silhouette graph
fig = silhouette_graph(X)

# Display the plot using Streamlit
st.pyplot(fig)


def kmeans(X):
    kmeans = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
    kmeans.fit(X)
    playlist_df['cluster'] = kmeans.predict(X)
    playlist_df['genres'] = playlist_df['genres']
    return playlist_df

playlist_df = kmeans(X)

# def tsne_graph(X):
#     tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
#     genre_embedding = tsne_pipeline.fit_transform(X)
#     projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
#     projection['genres'] = playlist_df['genres']
#     projection['cluster'] = playlist_df['cluster']

#     fig = px.scatter(
#         projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
#     return st.pyplot(fig)

def tsne_graph(X, playlist_df):
    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = playlist_df['genres']
    projection['cluster'] = playlist_df['cluster']

    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
    return fig

# Assuming you have the data 'X' and 'playlist_df' available

# Create a Streamlit app
st.title('t-SNE Visualization')
st.write('This app performs t-SNE dimensionality reduction and visualizes the data.')

# Generate the t-SNE graph
fig = tsne_graph(X, playlist_df)

# Display the plot using Streamlit
st.plotly_chart(fig)

# def get_audio_features(song_name, artist):
#     query = f"track:{song_name} artist:{artist}"
#     results = sp.search(q=query, type='track', limit=1)

#     if len(results['tracks']['items']) > 0:
#         track_id = results['tracks']['items'][0]['id']
#         track_features = sp.audio_features([track_id])
        
#         if len(track_features) > 0:
#             track_info = sp.track(track_id)
#             popularity = track_info['popularity']
            
#             numerical_features = {key: value for key, value in track_features[0].items() if isinstance(value, (int, float))}
#             numerical_features['popularity'] = popularity
#             return numerical_features
#     return None

# audio_feats = get_audio_features('Dynamite', 'BTS')

def get_audio_features(song_name, artist):
    query = f"track:{song_name} artist:{artist}"
    results = sp.search(q=query, type='track', limit=1)

    if len(results['tracks']['items']) > 0:
        track_id = results['tracks']['items'][0]['id']
        track_features = sp.audio_features([track_id])
        
        if len(track_features) > 0:
            track_info = sp.track(track_id)
            popularity = track_info['popularity']
            
            numerical_features = {key: value for key, value in track_features[0].items() if isinstance(value, (int, float))}
            numerical_features['popularity'] = popularity
            return numerical_features
    return [0]

# Create a Streamlit app
st.title('Audio Features Lookup')
st.write('Enter a song name and artist to get the audio features.')

# Get user input
song_name = st.text_input('Song Name')
artist = st.text_input('Artist')

if song_name and artist:
    # Get audio features
    audio_feats = get_audio_features(song_name, artist)

    if audio_feats:
        st.write('Audio Features:')
        audio_table = pd.DataFrame.from_dict(audio_feats, orient='index', columns=['Value'])
        st.table(audio_table)
    else:
        st.write('No audio features found for the given song and artist.')

def knn(k, X, y, audio_feats):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)

    num_feat = pd.DataFrame.from_dict(audio_feats, orient='index').T
    num_feat = num_feat[COLS]

    new_input_scaled = scaler.transform(num_feat.values.reshape(1, -1))

    distances, indices = knn.kneighbors(new_input_scaled)
    
    nearest_songs = playlist_df.iloc[indices[0]]

    print("Nearest Songs:")
    print(playlist_df['track_name'], playlist_df['artist_name'])
    return nearest_songs['genres'],  playlist_df['track_name'], playlist_df['artist_name']

genre, track_name, artist_name = knn(5, X, y, audio_feats)
print(genre, track_name, artist_name)



#--------- Streamlit  ---------