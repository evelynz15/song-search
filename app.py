from fastapi import FastAPI
from pydantic import BaseModel
from recommend_songs import recommend, vibe_search_text, search_song_options
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Song Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class SongOptionsRequest(BaseModel):
    song: str

class SongRequest(BaseModel):
    track_id: str

class VibeRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Song Recommender API is running"}

@app.post("/search/song")
def search_song(req: SongOptionsRequest):
    results = search_song_options(req.song)
    return results

@app.post("/recommend/song")
def recommend_by_song(req: SongRequest):
    results = recommend(req.track_id)
    print(results)
    return results

@app.post("/recommend/vibe")
def recommend_by_vibe(req: VibeRequest):
    results = vibe_search_text(req.query)
    print(results)
    return results