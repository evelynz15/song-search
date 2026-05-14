
import faiss
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# converts sentences into vectors
from sentence_transformers import SentenceTransformer
from build_index import index

model = SentenceTransformer("all-MiniLM-L6-v2")

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

VIBE_MAP = {
    "happy": {"valence": 0.9},
    "sad": {"valence": 0.2},
    "upbeat": {"energy": 0.85, "tempo": 130},
    "chill": {"energy": 0.3, "tempo": 90},
    "dance": {"danceability": 0.9},
    "party": {"energy": 0.9, "danceability": 0.9},
    "calm": {"energy": 0.2},
    "angry": {"energy": 0.95, "valence": 0.2},
}

GENRES = [
    "pop", "rock", "hip hop", "rap",
    "edm", "dance", "latin", "r&b"
]

# build new index or update
# csv_file = input("Enter dataset file: ")
# index("spotify_songs.csv")

# load data
index = faiss.read_index("songs.index")
df = pd.read_csv("songs_cleaned.csv")
vectors = np.load("vectors.npy")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# finds all songs with matching name and returns options
def search_song_options(song_name):
    # Find song index
    matches = df[
        df["track_name"]
        .str.lower()
        .str.strip()  
        == song_name.lower().strip()
    ]

    matches = matches[[
        "track_id",
        "track_name",
        "track_artist",
        "playlist_genre"
    ]].drop_duplicates()

    if len(matches) == 0:
        print("Song not found")
        return None

    return matches.to_dict(orient="records")

# recommendation function by track id 
def recommend(track_id, k=5):
    # Find song index
    match = df[df["track_id"] == track_id]

    if len(match) == 0:
        return []
    
    idx = match.index[0]

    query_vector = vectors[idx].reshape(1, -1).astype("float32")

    distances, indices = index.search(query_vector, k + 1)

    results = df.iloc[indices[0][1:]].copy()

    results["similarity"] = distances[0][1:]

    return results[[
        "track_id",
        "track_name",
        "track_artist",
        "playlist_genre",
        "similarity"
    ]].to_dict(orient="records")

# helper to parse query for vibe recommendation
def parse_vibe(query):
    params = {
        "valence": 0.5,
        "energy": 0.5,
        "danceability": 0.5,
        "tempo": 120
    }

    words = query.lower().split()

    for word in words:
        if word in VIBE_MAP:
            for key, value in VIBE_MAP[word].items():
                params[key] = value
    
    # print("PARAMS:")
    # print(params)
    return params

# helper to extract genre from query if present
def extract_genre(query):
    query = query.lower()

    for genre in GENRES:
        if genre in query:
            return genre

    return None

# recommends by vibe from text 
def vibe_search_text(query, k=5):

    genre_filter = extract_genre(query)

    # Convert text to audio parameters
    params = parse_vibe(query)

    vibe = np.array([[ 
        params["danceability"],
        params["energy"],
        -10,
        0.1,
        0.3,
        0.0,
        0.1,
        params["valence"],
        params["tempo"]
    ]])

    vibe_df = pd.DataFrame(vibe, columns=FEATURE_COLUMNS)
    vibe_scaled = scaler.transform(vibe_df)

    # Convert text to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    vibe_full = np.hstack([
        vibe_scaled * 2.0,        # audio parameters
        query_embedding * 0.5            # semantic signal
    ]).astype("float32")

    faiss.normalize_L2(vibe_full)

    distances, indices = index.search(vibe_full, 50)

    results = df.iloc[indices[0]].copy()
    results["similarity"] = distances[0]

    # apply genre filter if applicable
    if genre_filter:
        results = results[
            results["playlist_genre"].str.lower() == genre_filter
        ]

    # fallback if none present
    if len(results) == 0:
        print("No matches for that genre, showing all results")
        results = df.iloc[indices[0]].copy()
        results["similarity"] = distances[0]

    return results.head(k)[[
        "track_id",
        "track_name",
        "track_artist",
        "playlist_genre",
        "similarity"
    ]].to_dict(orient="records")

# recommendation based on vibe from numeric input
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

    vibe_df = pd.DataFrame(vibe, columns=[
        "danceability", "energy", "loudness",
        "speechiness", "acousticness",
        "instrumentalness", "liveness",
        "valence", "tempo"
    ])

    # Scale using same scaler
    vibe_scaled = scaler.transform(vibe_df)

    embedding_dim = vectors.shape[1] - vibe_scaled.shape[1]

    # Add zero embeddings for semantic part
    vibe_full = np.hstack([
        vibe_scaled * 2.0,
        np.zeros((1, embedding_dim))
    ]).astype("float32")

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