
import faiss
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# converts sentences into vectors
from sentence_transformers import SentenceTransformer 

# load data 
df = pd.read_csv("spotify_songs.csv")
df = df.fillna("")
df = df.drop_duplicates(subset=["track_name", "track_artist"])
print(df.head())
print(df.columns)

# select desired features
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

audio_data = df[FEATURE_COLUMNS].astype(float)

# scale data to be in the same range
scaler = StandardScaler()
audio_features = scaler.fit_transform(audio_data)

# create genre text
genre_text = (
    df["playlist_genre"].astype(str) + " " +
    df["playlist_subgenre"].astype(str)
).tolist()

# embed genre information
model = SentenceTransformer("all-MiniLM-L6-v2")

genre_embeddings = model.encode(
    genre_text,
    show_progress_bar=True,
    convert_to_numpy=True
)

# combine into one final vector for each song
final_vectors = np.hstack([
    audio_features * 2.0,
    genre_embeddings * 0.5
])
final_vectors = final_vectors.astype("float32")

# normalize to use cosine similarity (converts each to have magnitude 1)
# only direction is relevant
faiss.normalize_L2(final_vectors)

dimension = final_vectors.shape[1]

# build FAISS index
index = faiss.IndexFlatIP(dimension)
index.add(final_vectors)

print("FAISS index built:", index.ntotal, "songs")

# recommendation function
def recommend(song_name, k=5):
    # Find song index
    matches = df[df["track_name"].str.lower() == song_name.lower()]

    if len(matches) == 0:
        print("Song not found")
        return None

    idx = matches.index[0]

    query_vector = final_vectors[idx].reshape(1, -1)

    distances, indices = index.search(query_vector, k + 1)

    results = df.iloc[indices[0][1:]].copy()

    results["similarity"] = distances[0][1:]

    return results[[
        "track_name",
        "track_artist",
        "playlist_genre",
        "similarity",
        "track_popularity"
    ]]

# recommendation based on vibe
def vibe_search(valence=0.5, energy=0.5, danceability=0.5, tempo=120, k=5):

    # Create fake audio vector
    vibe = np.array([[ 
        danceability,
        energy,
        -10,  # avg loudness
        0.1,
        0.3,
        0.0,
        0.1,
        valence,
        tempo
    ]])

    # Scale using same scaler
    vibe_scaled = scaler.transform(vibe)

    # Add zero embeddings for semantic part
    vibe_full = np.hstack([
        vibe_scaled * 2.0,
        np.zeros((1, genre_embeddings.shape[1]))
    ])
    vibe_full = vibe_full.astype("float32")

    faiss.normalize_L2(vibe_full)

    distances, indices = index.search(vibe_full, k)

    results = df.iloc[indices[0]].copy()
    results["similarity"] = distances[0]

    return results[[
        "track_name",
        "track_artist",
        "playlist_genre",
        "similarity"
    ]]

if __name__ == "__main__":

    print("\n🎧 Similar songs:\n")
    print(recommend("Shape of You", k=5))

    print("\n🔥 Happy party vibe:\n")
    print(vibe_search(valence=0.9, energy=0.8, danceability=0.8))