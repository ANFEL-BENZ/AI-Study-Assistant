
from pathlib import Path
import fitz
import os

# Make Tesseract accessible to Python
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

# Locate the project folder
project_dir = Path(__file__).resolve().parent.parent

# Locate the Arabic PDF
pdf_path = (
    project_dir
    / "documents"
    / "dzexams-doc-3as-histoire-geographie.pdf"
)

# Open the PDF
document = fitz.open(pdf_path)

try:
    # Select the first page
    page = document[0]

    # Recognize Arabic text using OCR
    text_page = page.get_textpage_ocr(
        language="ara",
        dpi=200,
        full=True
    )

    # Retrieve the recognized text
    text = page.get_text(textpage=text_page)

    print("PDF:", pdf_path.name)
    print("Page: 1")
    print("\nArabic text recognized by OCR:\n")
    print(text[:3000])

    # Save the result for comparison
    output_path = project_dir / "data" / "arabic_ocr_test.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print("\nSaved to:", output_path)

finally:
    document.close()
