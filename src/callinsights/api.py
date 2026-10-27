"""FastAPI service: ingest a call transcript, analyse it, store it, and report by intent."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .analytics import CallAnalyzer
from .generate import corpus
from .store import CallStore


class CallIn(BaseModel):
    id: str = Field(pattern=r"^[\w-]{1,40}$")
    text: str = Field(min_length=10)


class Outcome(BaseModel):
    transferred: int = Field(ge=0, le=1)


def create_app(analyzer: CallAnalyzer | None = None, store: CallStore | None = None) -> FastAPI:
    analyzer = analyzer or CallAnalyzer().fit(corpus())
    store = store or CallStore()
    app = FastAPI(title="call-insights-api")

    @app.post("/calls")
    def ingest(call: CallIn):
        result = analyzer.analyze(call.text).to_dict()
        store.add(call.id, result)
        return {"id": call.id, **result}

    @app.get("/calls/{call_id}")
    def get_call(call_id: str):
        row = store.get(call_id)
        if row is None:
            raise HTTPException(404, "call not found")
        return row

    @app.put("/calls/{call_id}/outcome")
    def outcome(call_id: str, body: Outcome):
        if not store.set_outcome(call_id, body.transferred):
            raise HTTPException(404, "call not found")
        return {"id": call_id, "transferred": body.transferred}

    @app.get("/report")
    def report(threshold: float = 0.5):
        return store.report(threshold)

    return app


app = create_app()
