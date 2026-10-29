from pathlib import Path
from pypdf import PdfReader

documents_dir = Path("documents")
pdf_files = list(documents_dir.glob("*.pdf"))

print(f"Nombre de PDF trouvés : {len(pdf_files)}")

for pdf_path in pdf_files:
    print("\n" + "=" * 60)
    print(f"Fichier : {pdf_path.name}")

    try:
        reader = PdfReader(str(pdf_path))
        print(f"Nombre de pages : {len(reader.pages)}")

        total_chars = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            total_chars += len(text)

        print(f"Caractères extraits : {total_chars}")

        if total_chars == 0:
            print("Attention : aucun texte extrait. Un OCR pourrait être nécessaire.")
        else:
            print("Aperçu du texte :")
            first_page_text = reader.pages[0].extract_text() or ""
            print(first_page_text[:300])

    except Exception as error:
        print(f"Erreur de lecture : {error}")
