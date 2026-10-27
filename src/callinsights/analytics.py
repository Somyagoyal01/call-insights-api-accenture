"""Call analytics: PII masking, intent, sentiment, transfer risk and a short summary.

All models are small and run offline (scikit-learn). Transfer risk is predicted
from the conversation BEFORE any transfer line, i.e. what an agent-assist tool
would see in time to act, so the label never leaks into the features.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

NEG = {"ridiculous", "frustrating", "worried", "angry", "nobody", "terrible", "upset", "unacceptable", "never",
       "wrong", "annoyed", "disappointed", "worst", "again"}
POS = {"thanks", "thank", "great", "perfect", "appreciate", "makes", "sense", "fine", "ok", "works", "alright"}
_MEMBER = re.compile(r"\bMBR-\d{6,10}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
_TRANSFER = re.compile(r"\btransfer", re.I)


def mask_pii(text: str) -> str:
    """Mask member IDs and Indian mobile numbers before anything is stored."""
    return _PHONE.sub("[PHONE]", _MEMBER.sub("[MEMBER_ID]", text))


def before_transfer(text: str) -> str:
    """Keep only the turns before the agent offers a transfer."""
    out = []
    for line in text.splitlines():
        if line.startswith("Agent:") and _TRANSFER.search(line):
            break
        out.append(line)
    return "\n".join(out)


def customer_text(text: str) -> str:
    return " ".join(l.split(":", 1)[1] for l in text.splitlines() if l.startswith("Customer:"))


def sentiment(text: str) -> float:
    """Lexicon score in [-1, 1] over customer turns (negative = upset)."""
    words = re.findall(r"[a-z']+", customer_text(text).lower())
    neg = sum(w in NEG for w in words)
    pos = sum(w in POS for w in words)
    return 0.0 if neg + pos == 0 else round((pos - neg) / (pos + neg), 3)


def summarize(text: str, intent: str) -> str:
    cust = [l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("Customer:")]
    agent = [l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("Agent:")]
    first = cust[0] if cust else ""
    last_agent = agent[-1] if agent else ""
    return f"[{intent}] Customer: {first[:120]} | Outcome: {last_agent[:120]}"


@dataclass
class CallResult:
    intent: str
    intent_confidence: float
    sentiment: float
    transfer_risk: float
    summary: str
    masked_text: str

    def to_dict(self):
        return asdict(self)


class CallAnalyzer:
    """Fit once on labelled calls; analyse new calls one at a time."""

    def __init__(self):
        self.intent_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                                          LogisticRegression(max_iter=2000, C=4.0))
        self.risk_text = TfidfVectorizer(ngram_range=(1, 1), min_df=2)
        self.risk_model = GradientBoostingClassifier(random_state=0)
        self.intents_: list[str] = []

    def _risk_features(self, texts: list[str], intents: list[str]) -> np.ndarray:
        one_hot = np.array([[int(i == k) for k in self.intents_] for i in intents])
        sent = np.array([[sentiment(t)] for t in texts])
        n_cust = np.array([[sum(l.startswith("Customer:") for l in t.splitlines())] for t in texts])
        return np.hstack([one_hot, sent, n_cust])

    def fit(self, calls: list[dict]) -> "CallAnalyzer":
        texts = [before_transfer(mask_pii(c["text"])) for c in calls]
        self.intent_model.fit([customer_text(t) for t in texts], [c["intent"] for c in calls])
        self.intents_ = list(self.intent_model.classes_)
        X = self._risk_features(texts, [c["intent"] for c in calls])
        self.risk_model.fit(X, [c["transferred"] for c in calls])
        return self

    def analyze(self, text: str) -> CallResult:
        masked = mask_pii(text)
        pre = before_transfer(masked)
        probs = self.intent_model.predict_proba([customer_text(pre)])[0]
        k = int(np.argmax(probs))
        intent = self.intents_[k]
        risk = float(self.risk_model.predict_proba(self._risk_features([pre], [intent]))[0, 1])
        return CallResult(intent=intent, intent_confidence=round(float(probs[k]), 3), sentiment=sentiment(pre),
                          transfer_risk=round(risk, 3), summary=summarize(masked, intent), masked_text=masked)
