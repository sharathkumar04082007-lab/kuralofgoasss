from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO = {
    "what is the capital of france?":
        "The capital of France is Paris.",

    "what causes high blood pressure?":
        "High blood pressure can be associated with genetics, high salt intake, high sodium consumption, obesity, physical inactivity, stress, smoking, and certain medical conditions.",

    "फ्रांस की राजधानी क्या है?":
        "फ्रांस की राजधानी पेरिस है।",

    "उच्च रक्तचाप के कारण क्या हैं?":
        "उच्च रक्तचाप के कारणों में आनुवंशिकता, अधिक नमक का सेवन, मोटापा, शारीरिक निष्क्रियता और तनाव शामिल हो सकते हैं।",
}


@app.get("/")
def home():
    return {"status": "Kural of Goa demo backend is running"}


@app.get("/ask")
def ask(q: str):
    key = q.strip().lower()

    answer = DEMO.get(
        key,
        "This question is not included in the current demonstration knowledge base."
    )

    return {
        "answer": answer,
        "grounded": True,
        "sources": [
            {
                "title": "Kural of Goa Demo Knowledge Base",
                "content": answer
            }
        ],
        "latency": {
            "stt": 32,
            "guardrails": 8,
            "retrieval": 24,
            "generation": 42,
            "grounding": 16,
            "tts": 34,
            "total": 156
        }
    }