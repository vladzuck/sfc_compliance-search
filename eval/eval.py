import requests
import numpy as np
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
API_URL = "http://127.0.0.1:8000/query"

TEST_CASES = [
    {
        "question": "What is the definition of automated trading services?",
        "expected": "Automated trading services are electronic facilities that regularly make or accept offers to buy or sell securities, futures contracts, or OTC derivatives resulting in binding transactions."
    },
    {
        "question": "What are the financial resource requirements for ATS providers?",
        "expected": "ATS providers must have sufficient financial resources to properly perform their operations, functions, and obligations and manage risks prudently."
    },
    {
        "question": "What are the system integrity requirements for ATS?",
        "expected": "Electronic facilities must maintain high levels of reliability, availability, and security including adequate capacity and contingency arrangements."
    },
    {
        "question": "What surveillance requirements apply to ATS providers?",
        "expected": "Activity conducted via the ATS must be subject to proper surveillance by the provider, a regulatory authority such as the SFC, or another competent person."
    },
    {
        "question": "What are the record keeping requirements for ATS providers?",
        "expected": "Providers must maintain full records of operations including proper audit trails and keep regulatory authorities informed of material changes."
    },
    {
        "question": "What governance arrangements are required for ATS providers?",
        "expected": "Providers must have robust, well-defined, and transparent governance arrangements to oversee management and decision-making."
    },
    {
        "question": "What transparency requirements apply to ATS providers?",
        "expected": "Providers must provide transparency regarding their operations, products, transactional information including order processing and clearing settlement arrangements, and rules."
    },
    {
        "question": "How does the SFC apply requirements to overseas ATS providers?",
        "expected": "The SFC applies requirements using a pragmatic approach, adapting regulatory requirements on a case-by-case basis depending on whether operations are based in Hong Kong or overseas."
    },
]


def get_embedding(text: str) -> list[float]:
    result = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_eval():
    print("=== RAG Evaluation ===\n")
    results = []

    for i, test in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] {test['question'][:60]}...")

        response = requests.post(API_URL, json={
            "question": test["question"],
            "n_results": 3
        })
        generated_answer = response.json()["answer"]

        expected_embedding = get_embedding(test["expected"])
        generated_embedding = get_embedding(generated_answer)
        similarity = cosine_similarity(expected_embedding, generated_embedding)

        passed = similarity >= 0.7
        results.append(passed)

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} — similarity: {similarity:.2f}\n")

    accuracy = sum(results) / len(results) * 100
    print(f"=== Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Accuracy: {accuracy:.0f}%")


if __name__ == "__main__":
    run_eval()