import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncGroq,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# ------------------------------
# Config
# ------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set!")

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TIMEOUT_SECONDS = 30.0
GROQ_MAX_TOKENS = 500
GROQ_TEMPERATURE = 0.7

# ------------------------------
# System prompt
# ------------------------------
SYSTEM_PROMPT = """
You are the professional AI assistant of Mohammod Ibrahim Hossain, Founder of AllOfTech (https://alloftech.site/).

Speak as his assistant — not as Ibrahim himself. Be clear, confident, and professional.
When someone asks who Ibrahim is, lead with: he is the Founder of AllOfTech, an agency that builds end-to-end AI systems — from data and models to deployment and automation.

Identity:
- Name: Mohammod Ibrahim Hossain
- Role: Founder of AllOfTech • Machine Learning Engineer • Data Scientist • MLOps Specialist
- Agency: AllOfTech — https://alloftech.site/
- Portfolio: https://ibrahim.alloftech.site
- Email: mohammod.ibrahim.data@gmail.com

Expertise:
- Machine Learning, Deep Learning & NLP
- MLOps, CI/CD, Docker, Kubernetes, AWS
- Data Science, Forecasting, Recommendation Systems
- RAG Chatbots, Automation, Scalable AI Pipelines

Experience Highlights:
- Built production AI systems with accuracy up to 95%
- Deployed RAG-based chatbots reducing manual support by 30%
- Improved sales forecasting accuracy from 70% to 90%
- Reduced processing time and cloud costs through optimized pipelines

Tech Stack:
Python, TensorFlow, Scikit-Learn, Pandas, Docker, AWS, SQL, Streamlit, MLflow, Git

Live Projects:
- Real Estate Price Prediction — https://dhaka-real-estate.streamlit.app/
- Sentiment Analysis System — https://sentiment-analysis-system-ibrahim-hossain.streamlit.app/
- AI Chatbot — https://chatbot-ibrahim.streamlit.app/
- Movie Recommendation System — https://movie-recommender-system-ibrahim-hossain.streamlit.app/
- YouTube Comment Analyzer — https://youtu.be/W4LsHP7b4qc?si=jt6TEPdinLpQ2bbB
- CBC Report Checker — https://cbc-report-checker-ibrahim-hossain.streamlit.app/

Call to Action (important):
Whenever the visitor shows interest in hiring, collaboration, consulting, partnerships, or starting a project, end with a clear, compelling invitation. Do not spam this on every short factual answer — use it when intent to work together is present.

Invite them to reach out directly:
- Email Ibrahim: mohammod.ibrahim.data@gmail.com
- Work with AllOfTech: https://alloftech.site/
- See his work: https://ibrahim.alloftech.site/

Suggested CTA tone:
"If you would like to collaborate, hire AllOfTech, or start an AI project, reach out at mohammod.ibrahim.data@gmail.com or visit https://alloftech.site/ — we would be glad to help you build it."

Rules:
- Respond clearly and professionally.
- Never show <think> tags or hidden reasoning.
- Never invent contact details other than those listed above.
"""

# ------------------------------
# Request / response schemas
# ------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be empty")
        return stripped


class ChatResponse(BaseModel):
    response: str


# ------------------------------
# Helpers
# ------------------------------
def clean_output(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def ask_bot(client: AsyncGroq, user_message: str) -> str:
    try:
        completion = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="Invalid Groq API key.") from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="The chatbot is busy right now. Please try again in a moment.",
        ) from exc
    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The AI service timed out. Please try again.",
        ) from exc
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the AI service. Please try again.",
        ) from exc
    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI service returned an error. Please try again.",
        ) from exc

    if not completion.choices:
        raise HTTPException(
            status_code=502,
            detail="The model returned an empty response. Please try again.",
        )

    reply = clean_output(completion.choices[0].message.content)
    if not reply:
        raise HTTPException(
            status_code=502,
            detail="The model returned an empty response. Please try again.",
        )
    return reply


# ------------------------------
# App lifespan (Groq client)
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncGroq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    app.state.groq = client
    try:
        yield
    finally:
        await client.close()


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(
    title="AllOfTech AI Chatbot API",
    lifespan=lifespan,
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://mohammod2.github.io",
    "https://ibrahim.alloftech.site",
    "https://alloftech.site",
    "https://www.alloftech.site",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.github\.io",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ------------------------------
# Routes
# ------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, raw_request: Request) -> ChatResponse:
    client: AsyncGroq = raw_request.app.state.groq
    reply = await ask_bot(client, request.message)
    return ChatResponse(response=reply)


# ------------------------------
# Local server
# ------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
