
import json
from pathlib import Path

# 1. Locate the JSON file
project_dir = Path(__file__).resolve().parent.parent
json_path = project_dir / "data" / "chunks.json"

# 2. Load the JSON file
with json_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

# 3. Display basic information
print("Total chunks:", len(chunks))
print("First chunk keys:", list(chunks[0].keys()) if chunks else [])

# 4. Check for missing or empty text
empty_chunks = [
    i for i, chunk in enumerate(chunks)
    if not chunk.get("text", "").strip()
]

# 5. Check that source and page are present
missing_metadata = [
    i for i, chunk in enumerate(chunks)
    if not chunk.get("source") or not chunk.get("page")
]

print("Empty chunks:", len(empty_chunks))
print("Chunks with missing metadata:", len(missing_metadata))

# 6. Show the number of chunks per PDF
counts = {}

for chunk in chunks:
    source = chunk["source"]
    counts[source] = counts.get(source, 0) + 1

print("\nChunks per PDF:")
for source, count in counts.items():
    print(f"{source}: {count}")
