import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'product_id' not in df.columns or 'name' not in df.columns:
        raise ValueError("CSV must contain 'product_id' and 'name' columns.")
    return df

def generate_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> tuple[np.ndarray, SentenceTransformer]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    return np.array(embeddings).astype("float32"), model

def save_embeddings(ids: list, embeddings: np.ndarray, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump({"ids": ids, "embeddings": embeddings}, f)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product for normalized vectors
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def main():
    # Paths
    data_path = "./data/processed/merged_dataset_with_id.csv"
    faiss_index_path = "./data/vectorstores/faiss_index"
    embedding_store_path = "./data/vectorstores/embeddings.pkl"

    # Workflow
    df = load_data(data_path)
    texts = df['name'].fillna("").tolist()
    embeddings, _ = generate_embeddings(texts)
    
    save_embeddings(df['product_id'].tolist(), embeddings, embedding_store_path)
    
    index = build_faiss_index(embeddings)
    save_faiss_index(index, faiss_index_path)

    print(f"[✔] FAISS index built and saved with {index.ntotal} vectors.")

if __name__ == "__main__":
    main()
