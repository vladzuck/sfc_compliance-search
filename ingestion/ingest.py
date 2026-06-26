import os
import time
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
import chromadb


load_dotenv()
client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400

SFC_DOCUMENTS = [
    {
        "path": "data/pdfs/sfc_automated_trading.pdf",
        "title": "Guidelines for Automated Trading Services",
        "doc_id": "sfc_automated_trading"
    },
    {
        "path": "data/pdfs/sfc_fund_manager.pdf",
        "title": "Fund Manager Code of Conduct",
        "doc_id": "sfc_fund_manager"
    },
    {
        "path": "data/pdfs/sfc_reit.pdf",
        "title": "Code on Real Estate Investment Trusts",
        "doc_id": "sfc_reit"
    }
]


def read_pdf(path: str) -> str:
    print(f"  Reading: {path}")
    pdf = PdfReader(path)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    print(f"  Extracted {len(text):,} characters from {len(pdf.pages)} pages")
    return text


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def get_embedding(text: str) -> list[float]:
    result = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    time.sleep(1)
    return result.embeddings[0].values


def setup_chromadb() -> chromadb.Collection:
    print("\nSetting up ChromaDB...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="sfc_documents")
    print(f"  Collection ready: '{collection.name}'")
    return collection

def ingest_documents(collection: chromadb.Collection):
    total_chunks = 0

    for doc in SFC_DOCUMENTS:
        print(f"\nProcessing: {doc['title']}")

        # check first before doing any work
        existing = collection.get(ids=[f"{doc['doc_id']}_chunk_0"])
        if existing['ids']:
            print(f"  Already ingested, skipping.")
            continue

        text = read_pdf(doc["path"])
        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")

        print(f"  Embedding {len(chunks)} chunks...")
        embeddings = [get_embedding(chunk) for chunk in chunks]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{doc['doc_id']}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{
                "source": doc["title"],
                "doc_id": doc["doc_id"],
                "chunk_index": i,
                "url": doc["path"]
            } for i in range(len(chunks))]
        )
        total_chunks += len(chunks)
        print(f"  Stored {len(chunks)} chunks in ChromaDB")

    print(f"\n✓ Ingestion complete — {total_chunks} total chunks stored")


if __name__ == "__main__":
    print("=== SFC Document Ingestion ===\n")
    collection = setup_chromadb()
    ingest_documents(collection)
    print("\nRun the API server next: uvicorn api.main:app --reload")
