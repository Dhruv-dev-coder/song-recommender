import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
data = pd.read_csv('spotify_tracks.csv')

# Data Preprocessing
data['name'] = data['name'].str.lower()
data.columns = data.columns.str.strip()
data = data[data['name'] != data['genre']]
data['duration_s'] = (data['duration_ms'] / 1000).round()
data = data.drop(columns=['duration_ms'])

# Label encoding
label_encoder = LabelEncoder()
data['artist_encoded'] = label_encoder.fit_transform(data['artists'])
data['genre_encoded'] = label_encoder.fit_transform(data['genre'])
data['album_encoded'] = label_encoder.fit_transform(data['album'])

# Combine features into a single column
data['combined_features'] = data['genre_encoded'].astype(str) + ' ' + \
                            data['artist_encoded'].astype(str) + ' ' + \
                            data['album_encoded'].astype(str) + ' ' + \
                            data['popularity'].astype(str) + ' ' + \
                            data['duration_s'].astype(str) + ' ' + \
                            data['explicit'].astype(str)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
features_matrix = vectorizer.fit_transform(data['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(features_matrix, features_matrix)

# Create name to index mapping
name_to_index = pd.Series(data.index, index=data['name']).to_dict()


# Song recommendation function
def get_song_recommendations_by_name(song_name, num_recommendations=10, explicit=None, duration=None, popularity=None):
    if song_name not in name_to_index:
        return f"Song '{song_name}' not found in the dataset."

    song_index = name_to_index[song_name]
    similarity_scores = list(enumerate(cosine_sim[song_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_songs = []

    for i in sorted_scores[1:]:
        song = data.iloc[i[0]]

        if explicit is not None and song['explicit'] != explicit:
            continue

        if (duration is not None and abs(song['duration_s'] - duration) > 20):
            continue

        if (popularity is not None and abs(song['popularity'] - popularity) > 10):
            continue

        recommended_songs.append({
            'song_name': song['name'],
            'artist': song['artists'],
            'album': song['album'],
            'explicit': song['explicit'],
            'duration': song['duration_s'],
            'popularity': song['popularity'],
            'genre': song['genre']
        })

        if len(recommended_songs) >= num_recommendations:
            break

    if recommended_songs:
        return recommended_songs
    else:
        return f"No recommendations found for '{song_name}' based on the given preferences."


# Streamlit interface
st.title("Song Recommendation System")

# Song Name Input
song_name = st.text_input("Enter the song name:", value="my love mine all mine - acoustic instrumental")

# Filter inputs
explicit_input = st.selectbox("Explicit Content:", ["Any",True, False])
duration_input = st.slider("Song Duration (seconds):", 0, 500, 130)
popularity_input = st.slider("Popularity (1-100):", 0, 100, 60)

# Get recommendations on button press
if st.button("Get Recommendations"):
    explicit_filter = None if explicit_input == "Any" else explicit_input
    recommended_songs = get_song_recommendations_by_name(
        song_name,
        num_recommendations=10,
        explicit=explicit_filter,
        duration=duration_input,
        popularity=popularity_input
    )

    if isinstance(recommended_songs, list):
        st.write(f"Recommended songs for '{song_name}':")
        for idx, song in enumerate(recommended_songs, 1):  # Start numbering from 1
            st.write(f"{idx}.")
            st.write(f"**Song:** {song['song_name']}")
            st.write(f"**Artist:** {song['artist']}")
            st.write(f"**Album:** {song['album']}")
            st.write(f"**Explicit:** {song['explicit']}")
            st.write(f"**Duration:** {song['duration']}s")
            st.write(f"**Popularity:** {song['popularity']}")
            st.write(f"**Genre:** {song['genre']}")
    else:
        st.write(recommended_songs)
# Streamlit interface