from fastapi import FastAPI
from pydantic import BaseModel
from recommend_songs import recommend, vibe_search_text
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