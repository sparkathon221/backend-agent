import faiss
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/processed/merged_dataset_with_id.csv"
INDEX_PATH = BASE_DIR / "data/vectorstores/faiss_index"
EMBEDDINGS_PATH = BASE_DIR / "data/vectorstores/embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

class Recommender:
    def __init__(self):
        """Initialize with data, model and FAISS index"""
        self.df = pd.read_csv(DATA_PATH)
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
            self.id_map = data["ids"]
            # Add any other needed data from embeddings.pkl
            self.product_data = data.get("product_data", {})
    
    def embed_text(self, query: str) -> np.ndarray:
        """Convert text query to embedding vector"""
        emb = self.model.encode([query], normalize_embeddings=True)
        return np.array(emb).astype("float32")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for products using text query"""
        query_vector = self.embed_text(query)
        return self._search_with_vector(query_vector, top_k)
    
    def search_with_vector(self, vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search using existing vector (for image search)"""
        return self._search_with_vector(vector, top_k)
    
    def _search_with_vector(self, vector: np.ndarray, top_k: int) -> List[Dict]:
        """Internal search implementation"""
        D, I = self.index.search(vector, top_k)
        matched_ids = [self.id_map[i] for i in I[0]]
        results = (
            self.df[self.df['product_id'].isin(matched_ids)]
            .fillna("")
            .to_dict(orient='records')
        )
        return results
    
    def hybrid_search(self, text: Optional[str] = None, 
                     image_vector: Optional[np.ndarray] = None,
                     top_k: int = 10) -> List[Dict]:
        """Combine text and image search results"""
        if text and image_vector is not None:
            text_vector = self.embed_text(text)
            combined_vector = (text_vector + image_vector) / 2
            return self._search_with_vector(combined_vector, top_k)
        elif text:
            return self.search(text, top_k)
        elif image_vector is not None:
            return self.search_with_vector(image_vector, top_k)
        else:
            raise ValueError("Either text or image_vector must be provided")