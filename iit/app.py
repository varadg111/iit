import os
import json
import sqlite3
import numpy as np
import logging
import aiohttp
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.6
MAX_RESULTS = 10

# FastAPI setup
app = FastAPI(title="RAG Query API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel): 
    question: str
    image: Optional[str] = None  # Unused

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# DB connection
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Cosine similarity
def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    if np.all(v1 == 0) or np.all(v2 == 0): return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Get embedding from aipipe API
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")

    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": text}

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        return (await resp.json())["data"][0]["embedding"]
                    else:
                        msg = await resp.text()
                        logger.warning(f"Embedding error (attempt {attempt+1}): {msg}")
                        await asyncio.sleep(2 * (attempt + 1))
        except Exception as e:
            logger.error(f"Exception during embedding: {e}")
            await asyncio.sleep(2 * (attempt + 1))

    raise HTTPException(status_code=500, detail="Failed to get embedding")

# Similar content from DB
async def find_similar_content(query_embedding, conn):
    cursor = conn.cursor()
    results = []

    for table in ["discourse_chunks", "markdown_chunks"]:
        cursor.execute(f"SELECT * FROM {table} WHERE embedding IS NOT NULL")
        for row in cursor.fetchall():
            try:
                emb = json.loads(row["embedding"])
                sim = cosine_similarity(query_embedding, emb)
                if sim >= SIMILARITY_THRESHOLD:
                    url = row["url"] if "url" in row.keys() else row["original_url"] if "original_url" in row.keys() else ""
                    results.append({
                        "content": row["content"],
                        "url": url,
                        "similarity": sim
                    })
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")

    return sorted(results, key=lambda r: r["similarity"], reverse=True)[:MAX_RESULTS]

# Ask GPT via aipipe
async def generate_answer(question, docs):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")

    context = "\n\n".join([f"From {doc['url']}\n{doc['content']}" for doc in docs])
    prompt = f"""Answer based only on the context below. If not answerable, say so.

Context:
{context}

Question: {question}

Return format:
1. Your answer
2. Sources:
1. URL: [url], Text: [snippet]
"""

    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Helpful assistant"},
            {"role": "user", "content": prompt}
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                return (await resp.json())["choices"][0]["message"]["content"]
            else:
                raise HTTPException(status_code=resp.status, detail=await resp.text())

# API endpoint
@app.post("/query", response_model=QueryResponse)
async def query_kb(request: QueryRequest):
    conn = get_db_connection()
    try:
        embedding = await get_embedding(request.question)
        results = await find_similar_content(embedding, conn)
        if not results:
            return {"answer": "No relevant content found.", "links": []}
        answer = await generate_answer(request.question, results)
        return {
            "answer": answer,
            "links": [{"url": r["url"], "text": r["content"][:100]} for r in results]
        }
    finally:
        conn.close()

@app.get("/health")
async def health_check():
    return {"status": "ok", "api_key_set": bool(API_KEY)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
