# call-insights-api-accenture

*Personal project by Somya Goyal, built while applying to Accenture. Not affiliated with or endorsed by Accenture; uses synthetic and hand-written sample calls only (no real customer data).*

A small Python service for a health-insurance contact centre. It takes a call transcript, masks personal data (member IDs, phone numbers), classifies the intent, scores customer sentiment, predicts the **risk the call ends in a transfer** before the transfer happens, stores the result in SQL and serves an operations report over a REST API.

## Why this matters for Accenture

- **The problem.** Contact centres lose time and customer goodwill on calls that get passed between teams. A supervisor who can see "this prior-authorisation call is heading for a transfer" early can step in.
- **Who feels it.** Accenture Operations teams that run and automate client processes. The *AI/ML Computational Science Associate* posting (AIOC-S01664060) asks for Python, OOP, SQL, REST APIs, Git, and "automation and process improvement" with "data processing, reporting, and analytics". Accenture's Health Analytics contact-centre team (S&C GN CCAI posting) describes intent classification, sentiment analysis and work "to reduce call transfers".
- **What this shows.** An end-to-end, testable Python backend with object-oriented design, SQLite with a reporting query, and FastAPI endpoints with validation. It masks personal data before storage and is honest about how well the models work on wording they haven't seen.
- **Expected impact (estimate).** On the synthetic test split, flagging calls at risk ≥ 0.5 catches 63% of eventual transfers, and 66% of flagged calls really do transfer. In a real centre the value depends on what a supervisor can do in time, which this project does not measure.

## Demo

```bash
pip install -r requirements.txt
python run.py                                   # writes results/metrics.json
pytest -q                                       # 6 tests
uvicorn callinsights.api:app --app-dir src      # POST /calls, GET /calls/{id}, PUT /calls/{id}/outcome, GET /report
```

```bash
curl -X POST localhost:8000/calls -H 'content-type: application/json' \
  -d '{"id":"c1","text":"Customer: my doctor needs approval for knee surgery, member MBR-12345678\nAgent: It is under review.\nCustomer: this is ridiculous"}'
```

## How it works

- `analytics.py`
  - `mask_pii` replaces member IDs and Indian mobile numbers before anything is stored or modelled.
  - `before_transfer` cuts the transcript at the agent's transfer line, so the model never sees the answer it is predicting.
  - **Intent:** TF-IDF + logistic regression on the customer turns.
  - **Sentiment:** a small generic lexicon over customer turns, scored from -1 to 1.
  - **Transfer risk:** gradient boosting on intent, sentiment and number of customer turns.
  - **Summary:** extractive, made of the first customer line plus the outcome line.
- `store.py`: a `CallStore` class over SQLite with parameterised queries. The `/report` SQL groups calls by intent and reports average sentiment, predicted risk, actual transfer rate and calls flagged for a supervisor.
- `api.py`: FastAPI with Pydantic validation. Bad IDs are rejected with 422 and unknown calls return 404.
- `generate.py`: 1,200 synthetic calls across 6 intents. Upset customers transfer more often.

## Results (from `python run.py`)

| set | written | intent accuracy | transfer AUC | flag precision / recall @0.5 |
|---|---|---|---|---|
| synthetic test split (300) | generator | 1.00 | 0.78 | 0.66 / 0.63 |
| hand-written (10) | before the code | 0.90 | 1.00 | 0.80 / 1.00 |
| **blind (8)** | **after code freeze** | **0.63** | 0.80 | **0.43 / 1.00** |

### What the numbers say

- **Intent accuracy of 1.00 on synthetic data only shows the generator is easy.** On blind, real-sounding wording, accuracy drops to 5/8. "Thyroid medicine delivered every 3 months" was read as a claim question. So were "premium deducted two times" and "won't admit my father until you approve it". The vocabulary is too narrow, and that is the first thing to fix (sentence embeddings or an LLM classifier, calibrated on labelled calls).
- **Transfer risk caught all 3 blind transfers but also flagged 4 calm calls.** The upset-word list misses phrases like "fed up" and "can't wait". I chose a small generic lexicon on purpose, and did not tune it to the test calls.
- **The hand-written set was written before the code but is tiny (10 calls)**, so 1.00 AUC there means little. The blind set is the honest check.
- An earlier version of the sentiment word list had words taken from the hand-written calls. I removed them, because they made the score self-confirming.

## Limitations / next steps

- Synthetic text, not speech. Real calls need ASR (speech-to-text), and ASR errors would lower every number above.
- The lexicon sentiment should become a trained model. Intent should get embeddings plus a confidence threshold that routes unclear calls to a person.
- `/report` is per intent only. Adding per-agent and time-of-day views, and storing model versions, would make it an audit trail.
- No authentication on the API yet.

## Files

- `src/callinsights/analytics.py`, `store.py`, `api.py`, `generate.py`, `evaluate.py`; `run.py`
- `data/heldout_calls_handwritten.json` (written before the code), `data/blind_calls.json` (written after the freeze)
- `tests/test_calls.py`: 6 tests (PII masking, no label leakage, sentiment, SQL report, API round trip, evaluation)
