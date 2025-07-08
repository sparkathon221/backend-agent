# services/recommend.py
import faiss
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # goes up from services → app → root

DATA_PATH = BASE_DIR / "data/processed/merged_dataset_with_id.csv"
INDEX_PATH = BASE_DIR / "data/vectorstores/faiss_index"
EMBEDDINGS_PATH = BASE_DIR / "data/vectorstores/embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

class Recommender:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
            self.id_map = data["ids"]

    def embed_text(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], normalize_embeddings=True)
        return np.array(emb).astype("float32")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_vector = self.embed_text(query)
        D, I = self.index.search(query_vector, top_k)
        matched_ids = [self.id_map[i] for i in I[0]]
        results = self.df[self.df['product_id'].isin(matched_ids)].to_dict(orient='records')
        return results
