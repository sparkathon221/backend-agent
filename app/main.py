from fastapi import FastAPI, Query
from pydantic import BaseModel
from services.recommend import Recommender
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
recommender = Recommender()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def root():
    return {"status": "OK", "message": "Shopping Assistant API"}

@app.post("/recommend")
def recommend_products(req: QueryRequest):
    results = recommender.search(req.query, top_k=req.top_k)
    return {"results": results}
