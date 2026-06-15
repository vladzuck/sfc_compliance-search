import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

app = FastAPI(
    title="SFC Compliance Search",
    description="RAG API over SFC regulatory documents",
    version="1.0.0"
)

client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="sfc_documents")


class QueryRequest(BaseModel):
    question: str
    n_results: int = 3

@app.post("/query")
def query(request: QueryRequest):
    try:
        question_embedding = client_genai.models.embed_content(
            model="gemini-embedding-001",
            contents=request.question
        ).embeddings[0].values

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=request.n_results
        )

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = ""
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
            context += f"\n[Source {i+1}: {meta['source']}]\n{chunk}\n"

        prompt = f"""You are a compliance assistant for Hong Kong financial regulations.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I cannot find this in the provided documents."
Always cite which source you used.

Context:
{context}

Question: {request.question}

Answer:"""

        response = client_genai.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return {
            "answer": response.text,
            "sources": [m["source"] for m in metadatas],
            "chunks_used": len(chunks)
        }
    except Exception as e:
        print(f"ERROR: {e}")
        raise

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_in_db": collection.count()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)