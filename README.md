# StudyMate AI 

**A Local AI-Powered Study Assistant for PDF Documents**

StudyMate AI is a Retrieval-Augmented Generation (RAG) application that helps students understand and explore their course materials by asking questions about PDF documents in natural language.

Instead of relying only on a language model's general knowledge, the application retrieves relevant passages from the provided documents and uses them as context to generate answers.

The project combines semantic search, vector embeddings, FAISS, and a locally running Large Language Model (LLM) through Ollama.

##  Features

- **PDF-based question answering:** Ask questions about your course documents.
- **Retrieval-Augmented Generation (RAG):** Ground answers in retrieved document passages.
- **Semantic search:** Retrieve relevant content based on meaning rather than exact keyword matching.
- **Multilingual embeddings:** Use multilingual sentence embeddings for semantic retrieval.
- **Vector similarity search:** Use FAISS to search document embeddings efficiently.
- **Local LLM inference:** Generate answers using Ollama and the `llama3.2:3b` model.
- **Document processing:** Extract text from PDF files and split it into manageable chunks.
- **Source-aware retrieval:** Keep document metadata, including source filenames and page numbers, to help identify relevant passages.

##  Technologies

| Technology                              | Purpose                          |
| --------------------------------------- | -------------------------------- |
| Python                                  | Core application logic           |
| Streamlit                               | Web application interface        |
| Ollama                                  | Local LLM inference              |
| Llama 3.2 (3B)                          | Answer generation                |
| Sentence Transformers                   | Text embeddings                  |
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual semantic embeddings |
| FAISS                                   | Vector similarity search         |
| PyPDF                                   | PDF text extraction              |

##  How It Works

StudyMate AI follows a Retrieval-Augmented Generation pipeline.

1. **Document ingestion:** Load PDF documents and extract their text.
2. **Text chunking:** Split the extracted text into smaller, overlapping passages.
3. **Embedding generation:** Convert each passage into a numerical vector using a multilingual embedding model.
4. **Vector indexing:** Store the vectors in a FAISS index for similarity search.
5. **Question processing:** Convert the user's question into an embedding.
6. **Context retrieval:** Retrieve the passages most semantically similar to the question.
7. **Answer generation:** Provide the retrieved context to the local LLM to generate a response.

```text
PDF Documents
      |
      v
Text Extraction
      |
      v
Text Chunking
      |
      v
Embedding Generation
      |
      v
FAISS Vector Index
      |
      v
User Question
      |
      v
Semantic Retrieval
      |
      v
Retrieved Context
      |
      v
Local LLM (Ollama)
      |
      v
Generated Answer
```

##  Project Structure

```text
AI_Study_Assistant/
├── app.py
├── app_backup.py
├── requirements.txt
├── .gitignore
├── documents/
│   └── Your PDF files
├── data/
│   ├── chunks.json
│   └── faiss.index
└── src/
    ├── ask.py
    ├── chunking.py
    ├── embeddings.py
    ├── retrieval.py
    ├── main.py
    ├── check_chunks.py
    ├── test_arabic_extraction.py
    ├── test_arabic_ocr.py
    └── test_pdfs.py
```

Generated index files and personal PDF documents are intended to remain local and are excluded from Git.

##  Getting Started

### Prerequisites

- Python 3.12
- Conda or another Python environment manager
- Ollama
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Study-Assistant.git
cd AI-Study-Assistant
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate the environment

Using Conda:

```bash
conda create -n rag-study python=3.12 -y
conda activate rag-study
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the language model

Install Ollama from [ollama.com](https://ollama.com), then download the model:

```bash
ollama pull llama3.2:3b
```

Make sure the Ollama service is running before starting the application.

### 5. Add your PDF documents

Place your PDF files in the project's `documents/` directory.

The application must process the documents and build its vector index before retrieval can work. Follow the indexing workflow implemented in the project if the index has not yet been generated.

### 6. Run the application

From the project root:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal by Streamlit.

##  Configuration

The application uses a local Ollama service and a multilingual sentence embedding model.

Depending on the current configuration, you may need to ensure that:

- Ollama is accessible at `http://127.0.0.1:11434`.
- The required LLM is available locally.
- The embedding model can be loaded.
- The PDF files have extractable text.
- The vector index has been generated from the current documents.

If you add or replace documents, rebuild the index so the new content becomes available to retrieval.

##  Privacy and Local Execution

StudyMate AI is designed to run locally. Its LLM inference uses Ollama on the user's machine, rather than requiring a paid hosted LLM API.

Local execution can help reduce the need to upload study materials to external AI services. However, privacy also depends on the application's configuration and the behavior of its dependencies.

##  Limitations

- Answer quality depends on PDF text extraction, chunking, retrieval quality, and the language model.
- Scanned PDFs may require OCR before their content can be retrieved.
- The model can produce inaccurate answers even when relevant passages are retrieved.
- Retrieval may fail when the question is ambiguous or the relevant content is not indexed.
- Multilingual embeddings do not guarantee equally accurate retrieval or generation in every language.
- Answers should be checked against the original course materials, especially for technical or academic questions.

##  Future Improvements

- Improve retrieval quality through chunking experiments and similarity-threshold tuning.
- Add source citations and clearer page-level references to generated answers.
- Improve Arabic and French question answering.
- Evaluate retrieval and answer quality on a test dataset.
- Explore reranking and hybrid keyword-semantic retrieval.
- Improve the user interface and document management workflow.

##  Learning Objectives

This project provides practical experience with:

- Retrieval-Augmented Generation (RAG)
- Natural Language Processing (NLP)
- Text embeddings and semantic similarity
- Vector databases and information retrieval
- Large Language Models (LLMs)
- PDF processing and text chunking
- Building local AI applications with Python\


---

*Built as a learning project exploring practical applications of local AI, semantic search, and Retrieval-Augmented Generation.*
