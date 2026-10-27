from fastapi.testclient import TestClient

from callinsights.analytics import CallAnalyzer, before_transfer, mask_pii, sentiment
from callinsights.api import create_app
from callinsights.evaluate import run_all
from callinsights.generate import corpus
from callinsights.store import CallStore

ANALYZER = CallAnalyzer().fit(corpus(400, seed=9))


def test_pii_is_masked_before_storage():
    out = mask_pii("Customer: call 9876543210 or +91 9123456780, member MBR-88213077")
    assert "9876543210" not in out and "MBR-88213077" not in out
    assert out.count("[PHONE]") == 2 and "[MEMBER_ID]" in out


def test_transfer_line_never_reaches_the_model():
    text = "Customer: my claim is stuck\nAgent: Let me transfer you to a specialist team.\nCustomer: ok"
    assert "transfer" not in before_transfer(text).lower()


def test_sentiment_direction():
    assert sentiment("Customer: this is ridiculous and frustrating") < 0 < sentiment("Customer: perfect, thanks")


def test_store_and_report_sql():
    s = CallStore()
    s.add("a", {"intent": "id_card", "intent_confidence": 0.9, "sentiment": 0.5, "transfer_risk": 0.1, "summary": "", "masked_text": ""})
    s.add("b", {"intent": "id_card", "intent_confidence": 0.9, "sentiment": -0.5, "transfer_risk": 0.7, "summary": "", "masked_text": ""})
    assert s.set_outcome("b", 1) and not s.set_outcome("zzz", 1)
    row = s.report(0.5)[0]
    assert row["calls"] == 2 and row["flagged_for_supervisor"] == 1 and row["actual_transfer_rate"] == 1.0


def test_api_roundtrip():
    c = TestClient(create_app(ANALYZER, CallStore()))
    r = c.post("/calls", json={"id": "x1", "text": "Customer: I need a new member ID card, member MBR-12345678\nAgent: Ordered."})
    assert r.status_code == 200 and r.json()["intent"] == "id_card" and "MBR-" not in r.json()["masked_text"]
    assert c.get("/calls/x1").json()["id"] == "x1"
    assert c.get("/calls/nope").status_code == 404
    assert c.put("/calls/x1/outcome", json={"transferred": 1}).status_code == 200
    assert c.get("/report").json()[0]["calls"] == 1
    assert c.post("/calls", json={"id": "bad id!", "text": "Customer: hello there"}).status_code == 422


def test_all_evaluations_run():
    m = run_all()
    assert m["dev_synthetic"]["intent_acc"] >= 0.9
    assert {"heldout_calls_handwritten", "blind_calls"} <= set(m)
