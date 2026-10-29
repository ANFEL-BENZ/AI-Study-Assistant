
from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "faiss.index"

# 2. Choose a multilingual model for English, French, and Arabic
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    # Load the chunks created from our PDFs
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "chunks.json not found. Run chunking.py first."
        )

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    if not chunks:
        print("No chunks found. Check your PDF extraction.")
        return

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loaded {len(texts)} chunks.")

    # 3. Load the embedding model
    print("Loading multilingual embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Convert text chunks into numerical vectors
    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # 5. Build a FAISS index for similarity search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 6. Save the index and the chunk metadata
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    print("\nEmbeddings created successfully!")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Vector dimensions: {dimension}")
    print(f"FAISS index saved to: {INDEX_PATH}")


if __name__ == "__main__":
    main()
