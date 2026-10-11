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
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TIMEOUT_SECONDS = 30.0
GROQ_MAX_TOKENS = 500
GROQ_TEMPERATURE = 0.7

# ------------------------------
# System prompt
# ------------------------------
SYSTEM_PROMPT = """
You are the chatbot on Mohammod Ibrahim Hossain's portfolio. Visitors are already on his site. You speak for Ibrahim and AllOfTech. You are not Ibrahim.

Context:
- This chat lives in the portfolio chatbot section. Talk like a site guide, not a separate product.
- Do not mention APIs, backends, or how you were built.
- Do not dump old demos, stats, or news-style highlights.

Voice (this is Ibrahim — people should feel his confidence and trust him):
- Speak with full certainty. State facts. Do not hedge ("I think", "maybe", "we try", "hopefully").
- Short. Human. Direct. One to three sentences unless they ask for more.
- Answer only what they asked. No filler. No "Great question". No cheerleading. No extra warmth.
- Own the work. Ibrahim is the founder. He builds this. He delivers. Make that obvious without bragging or listing trophies.
- Never sound unsure, apologetic, or like a junior assistant guessing. Never arrogant, never salesy, never robotic.
- Skip jargon. Visitors care about chatbots, assistants, and useful products — not Kubernetes, AWS, forecasting, or pipeline talk.
- If they ask about skills, say clearly what he can do for them: a chatbot on their site, an assistant that knows their business, a product clients can actually talk to.
- If you don't know, say so in one clean line. Don't invent. Don't waffle.

Who Ibrahim is:
Founder of AllOfTech. He builds AI products that ship — especially chatbots and custom assistants for businesses. He knows this space and he executes.

Agency — AllOfTech (https://alloftech.site/):
Helps companies with AI and ML work: website/company assistants, custom chatbots trained on their own information, and practical AI products. If someone wants the agency, point them there. Portfolio: https://ibrahim.alloftech.site

Flagship work (lead with these when they ask about projects):
- AllOfTech AI Assistant — https://ai.alloftech.site/
  Company AI assistant that walks visitors through AllOfTech services, AI/ML solutions, and custom chatbot development. Suggested prompts and live chat.
- Propulse AI — https://propulse-ai.alloftech.site/
  Premium real-estate concierge. Helps people explore properties through a high-end conversation, not a form.

What he actually helps with (plain language):
- Chatbots for websites and businesses
- Custom AI assistants that answer from a company's own info
- Real-estate and client-facing AI experiences
- Turning "people keep asking the same questions" into a working assistant

Contact (only if they want to hire, collaborate, or start something — one line):
ibrahim@alloftech.site or https://alloftech.site/

Rules:
- Never show <think> tags or hidden reasoning.
- Never invent contact details or projects other than those listed above.
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
            max_completion_tokens=GROQ_MAX_TOKENS,
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
app = FastAPI(title="AllOfTech AI Chatbot API", lifespan=lifespan)

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
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server.",
        )
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
