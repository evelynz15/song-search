import faiss
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

# converts sentences into vectors
from sentence_transformers import SentenceTransformer 

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

model = SentenceTransformer("all-MiniLM-L6-v2")

def index(file):
    # load data 
    new_df = pd.read_csv(file)
    new_df = new_df.fillna("")
    new_df = new_df.drop_duplicates(subset=["track_name", "track_artist"])
    print(new_df.head())
    print(new_df.columns)

    # check if index exists
    if not os.path.exists("songs.index"):

        print("No index found, building from scratch")

        df = new_df.copy()

        # select desired features
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

        # save
        faiss.write_index(index, "songs.index")
        df.to_csv("songs_cleaned.csv", index=False)
        np.save("vectors.npy", final_vectors)

        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print("new FAISS index built:", index.ntotal, "songs")

    else:
        print("Existing index found, checking for updates")

        # load existing
        index = faiss.read_index("songs.index")
        df = pd.read_csv("songs_cleaned.csv")
        vectors = np.load("vectors.npy")

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # find new songs
        new_only = new_df[
            ~new_df["track_id"].isin(df["track_id"])
        ]

        if len(new_only) == 0:
            print("No new songs, index unchanged")
            return

        print(f"Found {len(new_only)} new songs")

        # process new songs
        new_audio = scaler.transform(new_only[FEATURE_COLUMNS])

        genre_text = (
            new_only["playlist_genre"].astype(str) + " " +
            new_only["playlist_subgenre"].astype(str)
        ).tolist()

        new_embeddings = model.encode(genre_text, convert_to_numpy=True)

        new_vectors = np.hstack([
            new_audio * 2.0,
            new_embeddings * 0.5
        ]).astype("float32")

        faiss.normalize_L2(new_vectors)

        # update index
        index.add(new_vectors)

        # update storage
        df = pd.concat([df, new_only], ignore_index=True)
        vectors = np.vstack([vectors, new_vectors])

        # save again
        faiss.write_index(index, "songs.index")
        df.to_csv("songs_cleaned.csv", index=False)
        np.save("vectors.npy", vectors)

        print("Index updated incrementally")
