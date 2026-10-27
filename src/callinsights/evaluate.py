"""Dev-split metrics on synthetic calls, plus the hand-written and blind held-out sets."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score

from .analytics import CallAnalyzer, mask_pii
from .generate import corpus

DATA = Path(__file__).resolve().parents[2] / "data"


def score(analyzer: CallAnalyzer, calls: list[dict]) -> dict:
    res = [analyzer.analyze(c["text"]) for c in calls]
    y_int = [c["intent"] for c in calls]
    y_tr = [c["transferred"] for c in calls]
    risk = [r.transfer_risk for r in res]
    out = {"n": len(calls), "intent_acc": round(accuracy_score(y_int, [r.intent for r in res]), 3)}
    if len(set(y_tr)) == 2:
        out["transfer_auc"] = round(roc_auc_score(y_tr, risk), 3)
        flagged = [r >= 0.5 for r in risk]
        tp = sum(f and y for f, y in zip(flagged, y_tr))
        out["flag_precision@0.5"] = round(tp / max(sum(flagged), 1), 3)
        out["flag_recall@0.5"] = round(tp / max(sum(y_tr), 1), 3)
    out["intent_errors"] = [(c["id"], c["intent"], r.intent) for c, r in zip(calls, res) if r.intent != c["intent"]][:10]
    return out


def run_all() -> dict:
    calls = corpus()
    train, test = calls[:900], calls[900:]
    analyzer = CallAnalyzer().fit(train)
    out = {"dev_synthetic": score(analyzer, test)}
    for name in ("heldout_calls_handwritten", "blind_calls"):
        p = DATA / f"{name}.json"
        if p.exists():
            out[name] = score(analyzer, json.loads(p.read_text()))
    sample = "Customer: call me on 9876543210, member MBR-12345678"
    out["pii_example"] = mask_pii(sample)
    return out
