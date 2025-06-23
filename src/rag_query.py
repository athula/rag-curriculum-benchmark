import pickle
from sentence_transformers import SentenceTransformer

# Load FAISS index and chunks
with open("src/rag_index.pkl", "rb") as f:
    index, chunks = pickle.load(f)

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_prompt(question, context):
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

def query_rag_prompt_only(question, top_k=3):
    q_embed = embedder.encode([question])
    _, I = index.search(q_embed, top_k)
    retrieved_chunks = "\n".join([chunks[i] for i in I[0]])
    prompt = build_prompt(question, retrieved_chunks)
    return prompt
