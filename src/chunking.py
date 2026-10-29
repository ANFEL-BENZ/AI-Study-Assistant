
from pathlib import Path
from pypdf import PdfReader
import json

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
DATA_DIR = PROJECT_DIR / "data"

# Chunk configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward, keeping some text from the previous chunk.
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_and_chunk_pdfs() -> list[dict]:
    """Extract text from all PDFs and preserve source metadata."""
    all_chunks = []

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        print(f"\nProcessing: {pdf_path.name}")

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            page_chunks = split_text(text)

            for chunk_number, chunk in enumerate(page_chunks, start=1):
                all_chunks.append({
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk_number": chunk_number,
                })

        print(f"Total pages: {len(reader.pages)}")

    return all_chunks


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_and_chunk_pdfs()

    output_path = DATA_DIR / "chunks.json"
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nPDF files processed: {len(list(DOCUMENTS_DIR.glob('*.pdf')))}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Saved to: {output_path}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0]["text"][:500])
        print(f"Source: {chunks[0]['source']}")
        print(f"Page: {chunks[0]['page']}")


if __name__ == "__main__":
    main()
