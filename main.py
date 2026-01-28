import os
import re
import json
import asyncio
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set! ")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="AllOfTech AI Chatbot API")

# ------------------------------
# Allow CORS for your frontend domain
# ------------------------------
origins = [
    "http://localhost:3000",  # for local testing
    "http://127.0.0.1:5500",  # optional
    "http://localhost:5500",  # for local testing with Live Server
    "https://mohammod2.github.io",  # GitHub Pages - NO trailing slash, NO path
    "https://*.github.io",  # Optional: allow all GitHub Pages subdomains
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Be explicit
    allow_headers=["Content-Type", "Authorization"],
)
# ------------------------------
# Chatbot personality
# ------------------------------
AGENCY_SYSTEM_PROMPT = """
Mohammod Ibrahim Hossain
Machine Learning Engineer • Data Scientist • MLOps Specialist

I build end-to-end AI systems — from data and models to deployment and automation.

Expertise:

Machine Learning, Deep Learning & NLP

MLOps, CI/CD, Docker, Kubernetes, AWS

Data Science, Forecasting, Recommendation Systems

RAG Chatbots, Automation, Scalable AI Pipelines

Experience Highlights:

Built production AI systems with accuracy up to 95%

Deployed RAG-based chatbots reducing manual support by 30%

Improved sales forecasting accuracy from 70% → 90%

Reduced processing time and cloud costs through optimized pipelines

Tech Stack:
Python, TensorFlow, Scikit-Learn, Pandas, Docker, AWS, SQL, Streamlit, MLflow, Git

Projects (Live):

Real Estate Price Prediction
https://dhaka-real-estate.streamlit.app/

Sentiment Analysis System
https://sentiment-analysis-system-ibrahim-hossain.streamlit.app/

AI Chatbot
https://chatbot-ibrahim.streamlit.app/

Movie Recommendation System
https://movie-recommender-system-ibrahim-hossain.streamlit.app/

YouTube Comment Analyzer
https://youtu.be/W4LsHP7b4qc?si=jt6TEPdinLpQ2bbB

CBC Report Checker
https://cbc-report-checker-ibrahim-hossain.streamlit.app/

Portfolio: https://mohammod2.github.io/Protfolio/

Email: mohammod.ibrahim.data@gmail.com

Rules:
- Respond clearly and professionally.
- Never show <think> or hidden reasoning.
"""

# ------------------------------
# Request schema
# ------------------------------
class ChatRequest(BaseModel):
    message: str

# ------------------------------
# Clean output
# ------------------------------
def clean_output(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ------------------------------
# Async OpenRouter request
# ------------------------------
async def ask_bot(user_message: str) -> str:
    loop = asyncio.get_event_loop()

    def sync_request():
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "liquid/lfm-2.5-1.2b-instruct:free",
            "messages": [
                {"role": "system", "content": AGENCY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 500
        }

        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    return await loop.run_in_executor(None, sync_request)

# ------------------------------
# API endpoint
# ------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    reply = await ask_bot(request.message)
    return {"response": clean_output(reply)}  # frontend expects "response" key

# ------------------------------
# Run server locally
# ------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
