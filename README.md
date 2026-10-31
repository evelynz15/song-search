# Song Search

A semantic song recommendation API that finds similar tracks using a hybrid of audio features and natural-language "vibe" descriptions - built on ~30,000 Spotify tracks.

## Why this project

Most "recommend similar songs" demos just do genre matching or nearest-neighbor on raw audio stats. This project combines both signals into a single vector space so recommendations account for how a song *sounds* (tempo, energy, danceability, etc.) and how it's *described* (genre/subgenre text, embedded semantically) - plus a free-text "vibe" search that lets a user type something like `"sad acoustic"` or `"chill latin"` and get relevant tracks without needing a seed song at all.

## How it works

1. **Feature engineering**: Nine audio features (danceability, energy, loudness, tempo, etc.) are pulled from the Spotify dataset and standardized with `scikit-learn`'s `StandardScaler`.
2. **Semantic embedding**: Genre and subgenre text is embedded with a `sentence-transformers` model (`all-MiniLM-L6-v2`), so songs with related-but-not-identical genres end up close together in vector space.
3. **Hybrid vector index**: Scaled audio features and genre embeddings are concatenated into one vector per song, L2-normalized, and indexed with `FAISS` (`IndexFlatIP`) for fast cosine-similarity search.
4. **Vibe search**: A free-text query is parsed into approximate audio targets (via a mood-to-feature mapping) *and* embedded semantically, so "angry" and "party" pull toward different parts of the vector space even without a seed song.

## Tech stack

| Layer | Tools |
|---|---|
| API | FastAPI, Uvicorn |
| Vector search | FAISS |
| ML / embeddings | sentence-transformers, scikit-learn |
| Data | pandas, numpy |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/search/song` | POST | Look up a track by name to get its `track_id` and disambiguate (e.g. remixes/covers). |
| `/recommend/song` | POST | Given a `track_id`, return the 5 most similar tracks. |
| `/recommend/vibe` | POST | Given a free-text mood/query, return the 5 best-matching tracks. |
