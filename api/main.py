import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SFC Compliance Search</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f1117;
            color: #e1e4e8;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 48px 24px;
        }
        .container { width: 100%; max-width: 720px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 28px; font-weight: 600; color: #ffffff; margin-bottom: 8px; }
        .header p { font-size: 15px; color: #8b949e; line-height: 1.6; }
        .badge {
            display: inline-block; font-size: 11px; padding: 3px 10px;
            border-radius: 999px; background: #1f6feb22; color: #58a6ff;
            border: 1px solid #1f6feb55; margin-top: 12px;
        }
        .search-box {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }
        textarea {
            width: 100%; background: #0d1117; border: 1px solid #30363d;
            border-radius: 8px; color: #e1e4e8; font-size: 15px;
            padding: 12px 16px; resize: vertical; min-height: 80px;
            font-family: inherit; outline: none; transition: border-color .2s;
        }
        textarea:focus { border-color: #58a6ff; }
        textarea::placeholder { color: #484f58; }
        .controls {
            display: flex; align-items: center;
            justify-content: space-between; margin-top: 12px;
        }
        .hint { font-size: 12px; color: #484f58; }
        button {
            background: #238636; color: #ffffff; border: none;
            border-radius: 8px; padding: 10px 24px; font-size: 14px;
            font-weight: 500; cursor: pointer; transition: background .2s;
        }
        button:hover { background: #2ea043; }
        button:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
        .result {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 12px; padding: 24px; display: none;
        }
        .result.visible { display: block; }
        .result-label {
            font-size: 11px; font-weight: 600; letter-spacing: .06em;
            text-transform: uppercase; color: #8b949e; margin-bottom: 12px;
        }
        .answer {
            font-size: 15px; line-height: 1.8; color: #e1e4e8;
            white-space: pre-wrap; margin-bottom: 20px;
        }
        .sources {
            border-top: 1px solid #21262d; padding-top: 16px;
            display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
        }
        .sources-label { font-size: 12px; color: #8b949e; }
        .source-chip {
            font-size: 12px; padding: 3px 10px; border-radius: 999px;
            background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb44;
        }
        .loading { text-align: center; padding: 32px; display: none; }
        .loading.visible { display: block; }
        .spinner {
            width: 24px; height: 24px; border: 2px solid #30363d;
            border-top-color: #58a6ff; border-radius: 50%;
            animation: spin .8s linear infinite; margin: 0 auto 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading p { font-size: 14px; color: #8b949e; }
        .examples { margin-bottom: 24px; }
        .examples-label { font-size: 12px; color: #8b949e; margin-bottom: 8px; }
        .example-chips { display: flex; gap: 8px; flex-wrap: wrap; }
        .chip {
            font-size: 12px; padding: 5px 12px; border-radius: 999px;
            background: #161b22; border: 1px solid #30363d; color: #8b949e;
            cursor: pointer; transition: all .2s;
        }
        .chip:hover { border-color: #58a6ff; color: #58a6ff; }
        .error {
            background: #2d1b1b; border: 1px solid #f8514922;
            border-radius: 12px; padding: 16px; color: #f85149;
            font-size: 14px; display: none;
        }
        .error.visible { display: block; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>SFC Compliance Search</h1>
        <p>Ask questions about SFC regulatory documents.<br>Answers are grounded in official guidelines with cited sources.</p>
        <span class="badge">Powered by RAG · Gemini · ChromaDB</span>
    </div>

    <div class="examples">
        <div class="examples-label">Try an example</div>
        <div class="example-chips">
            <span class="chip" onclick="fillQuestion(this)">What are the requirements for automated trading services?</span>
            <span class="chip" onclick="fillQuestion(this)">What governance arrangements are required for ATS providers?</span>
            <span class="chip" onclick="fillQuestion(this)">How does the SFC apply requirements to overseas providers?</span>
        </div>
    </div>

    <div class="search-box">
        <textarea id="question" placeholder="Ask a question about SFC regulations..."></textarea>
        <div class="controls">
            <span class="hint">Press Enter to search</span>
            <button id="btn" onclick="search()">Search</button>
        </div>
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>Searching regulatory documents...</p>
    </div>

    <div class="error" id="error"></div>

    <div class="result" id="result">
        <div class="result-label">Answer</div>
        <div class="answer" id="answer"></div>
        <div class="sources">
            <span class="sources-label">Sources:</span>
            <span class="source-chip" id="source"></span>
        </div>
    </div>
</div>

<script>
function fillQuestion(el) {
    document.getElementById('question').value = el.textContent;
}

document.getElementById('question').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        search();
    }
});

async function search() {
    const question = document.getElementById('question').value.trim();
    if (!question) return;

    const btn = document.getElementById('btn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const error = document.getElementById('error');

    btn.disabled = true;
    loading.classList.add('visible');
    result.classList.remove('visible');
    error.classList.remove('visible');

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, n_results: 3 })
        });

        if (!response.ok) throw new Error('Server error');
        const data = await response.json();

        document.getElementById('answer').textContent = data.answer;
        document.getElementById('source').textContent = [...new Set(data.sources)].join(', ');

        result.classList.add('visible');
    } catch (err) {
        error.textContent = 'Something went wrong. Please try again.';
        error.classList.add('visible');
    } finally {
        btn.disabled = false;
        loading.classList.remove('visible');
    }
}
</script>
</body>
</html>
"""


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
            model="gemini-3-flash-preview",
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