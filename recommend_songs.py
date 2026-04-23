
import faiss
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# converts sentences into vectors
from sentence_transformers import SentenceTransformer
from build_index import index 

# build new index or update
csv_file = input("Enter dataset file: ")
index(csv_file)

# load data
index = faiss.read_index("songs.index")
df = pd.read_csv("songs_cleaned.csv")
vectors = np.load("vectors.npy")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# recommendation function
def recommend(song_name, k=5):
    # Find song index
    matches = df[df["track_name"].str.lower() == song_name.lower()]

    if len(matches) == 0:
        print("Song not found")
        return None

    idx = matches.index[0]

    query_vector = vectors[idx].reshape(1, -1).astype("float32")

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

if __name__ == "__main__":

    while True:
        print("Song Recommender")
        print("1. Find similar songs")
        print("2. Search by vibe")
        print("3. Exit")

        choice = input("\nEnter choice (1/2/3): ").strip()

        # song recommendation
        if choice == "1":
            song = input("\nEnter song name: ").strip()

            results = recommend(song, k=5)

            if results is not None:
                print("\nSimilar songs:\n")
                print(results.to_string(index=False))

        # vibe search
        elif choice == "2":
            try:
                print("\nEnter vibe values (0 → 1):")

                valence = float(input("Happiness: "))
                energy = float(input("Energy: "))
                danceability = float(input("Danceability: "))

                tempo_input = input("Tempo (default 120): ").strip()
                tempo = float(tempo_input) if tempo_input else 120

                results = vibe_search(
                    valence=valence,
                    energy=energy,
                    danceability=danceability,
                    tempo=tempo,
                    k=5
                )

                print("\nVibe results:\n")
                print(results.to_string(index=False))

            except ValueError:
                print("\nInvalid input — please enter numbers")

        # exit
        elif choice == "3":
            print("\nGoodbye")
            break

        else:
            print("\nInvalid choice")