
from pathlib import Path
import fitz

# 1. Locate the project and the Arabic PDF
project_dir = Path(__file__).resolve().parent.parent
pdf_path = (
    project_dir
    / "documents"
    / "dzexams-doc-3as-histoire-geographie.pdf"
)

# 2. Open the PDF
document = fitz.open(pdf_path)

# 3. Extract text from the first page
page = document[0]
text = page.get_text("text")

# 4. Display the extracted text
print("PDF:", pdf_path.name)
print("Page: 1")
print("\nExtracted Arabic text:\n")
print(text[:2000])

# 5. Close the PDF
document.close()
