
from pathlib import Path
import json
import urllib.request
import urllib.error

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "faiss.index"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LLM_MODEL = "llama3.2:3b"

OLLAMA_URL = "http://localhost:11434/api/generate"


def retrieve_chunks(question, model, index, chunks, top_k=3):
    """Find the most relevant PDF chunks for a question."""

    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_vector = np.asarray(
        question_vector, dtype="float32"
    )

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": float(score),
        })

    return results


def generate_answer(question, passages):
    """Ask Ollama to answer using retrieved PDF passages."""

    context_parts = []

    for i, passage in enumerate(passages, start=1):
        context_parts.append(
            f"[Source {i}: {passage['source']}, "
            f"page {passage['page']}]\n"
            f"{passage['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI Study Assistant.

Answer the student's question using the provided PDF passages.
The passages may be in English, French, or Arabic.
Answer in the same language as the student's question.
Explain clearly and simply.
Do not invent facts that are not supported by the passages.
If the passages do not contain enough information, say so.

PDF PASSAGES:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


def main():
    # Check the required files
    if not CHUNKS_PATH.exists():
        print("chunks.json is missing. Run chunking.py first.")
        return

    if not INDEX_PATH.exists():
        print("faiss.index is missing. Run embeddings.py first.")
        return

    # Load chunks, index, and embedding model
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nAI Study Assistant is ready!")
    print("Ask a question, or type 'exit' to stop.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            print("\nSearching your PDFs...")
            passages = retrieve_chunks(
                question, embedding_model, index, chunks
            )

            if not passages:
                print("No relevant passages were found.")
                continue

            print("Generating an answer with Ollama...")
            answer = generate_answer(question, passages)

            print("\n--- Answer ---")
            print(answer)

            print("\n--- Sources ---")
            seen = set()

            for passage in passages:
                citation = (
                    passage["source"], passage["page"]
                )

                if citation not in seen:
                    seen.add(citation)
                    print(
                        f"- {passage['source']}, "
                        f"page {passage['page']}"
                    )

        except urllib.error.URLError as error:
            print(
                "\nCould not connect to Ollama. "
                "Make sure Ollama is running."
            )
            print(f"Details: {error}")

        except Exception as error:
            print(f"\nSomething went wrong: {error}")


if __name__ == "__main__":
    main()
