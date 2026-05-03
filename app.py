from fastapi import FastAPI
from pydantic import BaseModel
from recommend_songs import recommend, vibe_search_text

app = FastAPI(title="Song Recommender API")
# -----------------------------
# Request models
# -----------------------------
class SongRequest(BaseModel):
    song: str

class VibeRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Song Recommender API is running"}

@app.post("/recommend/song")
def recommend_by_song(req: SongRequest):
    results = recommend(req.song)
    print(results)
    return results

@app.post("/recommend/vibe")
def recommend_by_vibe(req: VibeRequest):
    results = vibe_search_text(req.query)
    print(results)
    return results