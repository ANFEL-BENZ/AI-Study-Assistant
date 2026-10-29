
from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "faiss.index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def search_documents(question: str, top_k: int = 3):
    # Load chunks and metadata
    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    # Load the saved FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # Load the same model used to create the embeddings
    model = SentenceTransformer(MODEL_NAME)

    # Convert the question into a vector
    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Search for the most similar chunks
    scores, indices = index.search(
        question_vector.astype("float32"),
        min(top_k, len(chunks)),
    )

    # Display the results
    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx < 0:
            continue

        chunk = chunks[idx]

        print(f"\n--- Result {rank} ---")
        print(f"Similarity score: {score:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print(f"Text: {chunk['text'][:700]}")


if __name__ == "__main__":
    user_question = input("Ask a question about your PDFs: ")
    search_documents(user_question)
