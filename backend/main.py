"""Phishing email detector: FastAPI backend around cybersectony/phishing-email-detection-distilbert_v2.4.1."""
#imports
import re
from contextlib import asynccontextmanager
from email import policy
from email.parser import BytesParser
from pathlib import Path

import torch
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
LABELS = ["legitimate_email", "phishing_email", "legitimate_url", "phishing_url"]
EMAIL_PAIR = (0, 1)  # (legitimate_email, phishing_email): used for the email text
URL_PAIR = (2, 3)    # (legitimate_url, phishing_url): used for each URL
MAX_URLS = 25
MAX_EML_BYTES = 5 * 1024 * 1024
HIGH, MEDIUM = 0.70, 0.40  # verdict thresholds on the email phishing score
LINK_FLAG = 0.50           # a URL at or above this phishing score is flagged as suspicious

URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+", re.I)
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID)
    ml["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    yield
    ml.clear()

#production would have https encryption and authentication, but for this demo we just serve the frontend and API without auth
app = FastAPI(title="Phishing detector", lifespan=lifespan)


class TextIn(BaseModel):
    subject: str = Field(default="", max_length=1000)
    text: str = Field(min_length=1, max_length=200_000)


def classify(texts: list[str], pair: tuple[int, int]) -> list[dict]:
    """Score texts using only the (legitimate, phishing) class pair, renormalized to sum to 1."""
    legit_i, phish_i = pair
    inputs = ml["tokenizer"](
        texts, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.inference_mode():
        probs = torch.softmax(ml["model"](**inputs).logits, dim=-1)
    out = []
    for row in probs.tolist():
        legit, phish = row[legit_i], row[phish_i]
        score = phish / ((legit + phish) or 1e-9)
        out.append({
            "label": LABELS[phish_i] if score >= 0.5 else LABELS[legit_i],
            "phishing_score": score,
            "probs": {LABELS[legit_i]: 1 - score, LABELS[phish_i]: score},
        })
    return out


def extract_urls(text: str, hrefs: list[str]) -> list[str]:
    found = [u.rstrip(".,;:!?") for u in URL_RE.findall(text)] + hrefs
    return list(dict.fromkeys(found))[:MAX_URLS]  # dedupe, keep order


def build_email_input(subject: str, urls: list[str], body: str) -> str:
    # Subject and URLs go before the body so they survive the 512-token truncation.
    parts = [subject.strip(), "\n".join(urls), body.strip()]
    return "\n\n".join(p for p in parts if p)


def verdict_for(score: float) -> str:
    if score >= HIGH:
        return "likely_phishing"
    if score >= MEDIUM:
        return "suspicious"
    return "likely_legitimate"


def analyze(subject: str, body: str, hrefs: list[str] | None = None, headers: dict | None = None) -> dict:
    urls = extract_urls(body, hrefs or [])
    email_result = classify([build_email_input(subject, urls, body)], EMAIL_PAIR)[0]
    links = [{"url": u, **r} for u, r in zip(urls, classify(urls, URL_PAIR))] if urls else []
    for l in links:
        l["suspicious"] = l["phishing_score"] >= LINK_FLAG
    score = email_result["phishing_score"]  # links are flagged, not scored
    return {
        "verdict": verdict_for(score),
        "score": score,
        "suspicious_links": sum(l["suspicious"] for l in links),
        "email": email_result,
        "links": links,
        "headers": headers or {},
    }


def parse_eml(raw: bytes) -> tuple[str, str, list[str], dict]:
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    headers = {k: str(msg[k]) for k in ("From", "To", "Subject", "Date", "Reply-To", "Return-Path") if msg[k]}

    plain = msg.get_body(preferencelist=("plain",))
    html = msg.get_body(preferencelist=("html",))
    hrefs, text = [], ""

    if html is not None:
        soup = BeautifulSoup(html.get_content(), "html.parser")
        hrefs = [a["href"].strip() for a in soup.find_all("a", href=True)
                 if a["href"].lower().startswith(("http://", "https://"))]
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
    if plain is not None:  # prefer the plain-text part for the model input
        text = plain.get_content().strip()
    return headers.get("Subject", ""), text, hrefs, headers

#for actual production would have rate limiting i.e. slowAPI
#@limiter.limit("5/minute") etc
@app.post("/api/analyze")
def analyze_text(body: TextIn):
    return analyze(body.subject, body.text.strip())


@app.post("/api/analyze-eml")
def analyze_eml(file: UploadFile = File(...)):
    raw = file.file.read(MAX_EML_BYTES + 1)
    if len(raw) > MAX_EML_BYTES:
        raise HTTPException(413, "File is larger than 5 MB.")
    try:
        subject, text, hrefs, headers = parse_eml(raw)
    except Exception:
        raise HTTPException(422, "Could not parse this file as an .eml message.")
    if not text:
        raise HTTPException(422, "No readable body found in this message.")
    return analyze(subject, text, hrefs, headers)


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")