import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

with open("data/charlottes_web.txt") as f:
    text = f.read()

CHUNK_SIZE = 100
chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

with open("src/rag_index.pkl", "wb") as f:
    pickle.dump((index, chunks), f)

print("FAISS index built and saved.")

