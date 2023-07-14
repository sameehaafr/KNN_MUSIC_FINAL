#--------- Imports  ---------

import spotipy
import warnings
import pandas as pd
import webbrowser as wb
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

def X_y_split(playlist_df):
    X = playlist_df[COLS]
    y = playlist_df['genres']
    return X,y

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

def kmeans(X):
    kmeans = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=5))])
    kmeans.fit(X)
    playlist_df['cluster'] = kmeans.predict(X)
    playlist_df['genres'] = playlist_df['genres']
    return playlist_df

def tsne_graph(X, playlist_df):
    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = playlist_df['genres']
    projection['cluster'] = playlist_df['cluster']

    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
    return fig

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
    return None

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

    # print("Nearest Songs:")
    # print(nearest_songs['track_name'], nearest_songs['artist_name'])
    
    return nearest_songs['genres'], nearest_songs['track_name'], nearest_songs['artist_name']

#--------- STREAMLIT APP  ---------#
# Create a sidebar
st.sidebar.title("About Me")
# Add circular image
st.sidebar.markdown(
    """
    <style>
    .circle-image {
        border-radius: 50%;
        overflow: hidden;
        width: 150px;
        height: 150px;
    }
    </style>
    """
    , unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="circle-image">
        <img src="IMG_9670_1_2_50.png" alt="img">
    </div>
    """
    , unsafe_allow_html=True
)
st.sidebar.write("I'm Sameeha Afrulbasha! I'm an undergraduate student studying Data Science, Statistics, and Math at Purdue University. Feel free to checkout my website and other media accounts below!")

def open_link(url):
    js_code = f"window.open('{url}')"
    st.write(f"<script>{js_code}</script>", unsafe_allow_html=True)

if st.sidebar.button("My Website"):
    open_link("https://sameehaafr.github.io/sameehaafr/")
if st.sidebar.button("GitHub"):
    open_link("https://github.com/sameehaafr")
if st.sidebar.button("LinkedIn"):
    open_link("https://www.linkedin.com/in/sameeha-afrulbasha/")
if st.sidebar.button("Medium"):
    open_link("https://sameehaafr.medium.com/")
    


# Main content of the app
st.write("This is the main content of the app.")


st.title("Spotify Song Recommendation System Using KNN")
st.markdown("In this article, we will explore how to build a Spotify song recommendation system using machine learning techniques. By leveraging the Spotipy library and Spotify's Web API, we can retrieve audio features and genre information for songs from curated playlists. Through data analysis, clustering, and dimensionality reduction, we will uncover patterns in the music data.")
#st.write("Category: Unsupervised Machine Learning (K-Means and KNN)")
st.markdown("<b>Category:</b> Unsupervised Machine Learning (K-Means and KNN)", unsafe_allow_html=True)
st.markdown("<b>Objective:</b> Cluster data using K-means and build a KNN algorithm to recommend songs based on a users song input.", unsafe_allow_html=True)

st.header("Setting up the Environment")
st.markdown("To begin, we need to import the necessary libraries and set up the [Spotify](https://developer.spotify.com/) client credentials. We import Spotipy for interacting with the Spotify API, as well as other libraries such as pandas, streamlit, plotly, and scikit-learn for data processing and visualization. We also define the client ID and client secret provided by Spotify.")

# --------- READING DATA --------- #
st.header('Retrieving Spotify Data')
st.markdown("The code retrieves data from Spotify playlists using the Spotify Web API and the Spotipy library. A function called retrieve_spotify_data is defined, which takes an array of playlist URIs as input. For each playlist, the code extracts track information, audio features, and genres. The data is stored in a pandas DataFrame and saved as a CSV file for further analysis. The bigger the dataset the better. However, for this simple application I decided to do a playlist for each genre: Pop, Hip-Hop, Country, Latin, Rock, EDM, Indie, R&B, Jazz, and Metal. I will be expanding this soon. Here is the retrieved dataset with less important features removed.")
playlist_df = pd.read_csv('data/music_data.csv')
st.dataframe(playlist_df)

# --------- DATA PROCESSING --------- #
st.header('Data Processing and Feature Engineering')
st.markdown("Next, the code performs data processing and feature engineering tasks on the retrieved Spotify data. The DataFrame is split into input features (X) and the target variable (genres) (y) for classification purposes. Less important features are removed, while important features related to the essence of music, such as acousticness, danceability, energy, instrumentalness, loudness, speechiness, tempo, and valence, are retained. We will next be performing K-Means Clustering on the data to group similar songs together based on their audio features.")
X, y = X_y_split(playlist_df)

# --------- SILHOUETTE ANALYSIS --------- #
st.header('Silhouette Analysis')
st.markdown("The code includes a silhouette analysis to evaluate the quality of the clustering results. The silhouette score is calculated for different values of K (the number of clusters) in K-means clustering. The resulting plot displays the relationship between the number of clusters and the quality of the clustering. A higher silhouette score indicates better separation between clusters, aiding in the determination of the optimal number of clusters for K-means clustering. As the graph below shows, the optimal number of clusters is around 5-6.")
fig = silhouette_graph(X)
st.pyplot(fig)

# --------- KMEANS CLUSTERING --------- #
st.header('K-Means Clustering')
st.markdown("K-Means is a clustering algorithm used in unsupervised machine learning. It partitions a dataset into K distinct clusters where each data point belongs to the cluster with the nearest mean (centroid).")
st.markdown("In this section, K-means clustering is applied to the input features (audio features) using the optimized K value (5 or 6) from the silhouette analysis to group similar tracks together. The K-means algorithm assigns cluster labels to each data point based on their audio feature values. The resulting cluster labels are added as a new column to the DataFrame, enabling further analysis and visualization of the clustered data.")
playlist_df = kmeans(X)

# --------- t-SNE VISUALIZATION --------- #
st.header('t-SNE Visualization')
st.markdown("To visualize the clustered data, the code uses t-SNE (t-Distributed Stochastic Neighbor Embedding) for dimensionality reduction. t-SNE maps high-dimensional data to a lower-dimensional space while preserving local and global structure. The resulting 2D scatter plot displays the relationships between tracks, clusters, and genres, providing insights into the distribution and patterns within the data.")
fig = tsne_graph(X, playlist_df)
st.plotly_chart(fig)

# --------- KNN RECOMMENDER --------- #
st.header('KNN Recommender')
st.markdown("Now it is time to implement the K-nearest neighbors (KNN) recommender system. Users can input a song name and artist to receive personalized recommendations. The code retrieves the audio features for the input song using the Spotify API. If the audio features are found, they are displayed in a table, providing insights into the characteristics of the input song. The KNN algorithm utilizes the input audio features and the previously clustered data to find the nearest songs in terms of audio feature similarity. The recommended genres, tracks, and artists are displayed in a table, offering personalized recommendations based on user input.")
song_name = st.text_input('Song Name', 'Crazy in Love')
artist = st.text_input('Artist', 'Beyonce')
audio_feats = None
if song_name and artist:
    audio_feats = []
    audio_feats = get_audio_features(song_name, artist)

    if audio_feats:
        st.write('Audio Features:')
        audio_table = pd.DataFrame.from_dict(audio_feats, orient='index', columns=['Value'])
        st.table(audio_table)
    else:
        st.write('No audio features found for the given song and artist.')

genres, track, artist = knn(5, X, y, audio_feats)

# --------- RECOMMENDATIONS --------- #

st.write('Recommendations:')
recommendations_table = pd.DataFrame({'Genres': genres, 'Track': track, 'Artist': artist})
st.table(recommendations_table)

# --------- CONCLUSION --------- #

st.header('Conclusion')
st.markdown("By following this code and understanding the described steps, one can build a Spotify song recommendation system that utilizes audio features, clustering, and dimensionality reduction techniques to generate personalized song recommendations.")

# -------------ABOUT ME------------ #
st.header("Shameless Self-Promotion")
st.markdown("If you liked this project, checkout my other projects and some of my other media accounts below!")
st.markdown("Website: https://sameehaafr.github.io/sameehaafr/")
st.markdown("GitHub: https://github.com/sameehaafr")
st.markdown("LinkedIn: https://www.linkedin.com/in/sameeha-afrulbasha/")
st.markdown("Medium: https://sameehaafr.medium.com/")

#st.markdown("[Website](https://sameehaafr.github.io/sameehaafr/) \n\n\n [GitHub](https://github.com/sameehaafr) \n\n\n [LinkedIn]() \n\n\n [Medium](https://sameehaafr.medium.com/)")

st.write("Thanks for reading! :)") 