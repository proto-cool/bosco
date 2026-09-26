"""The Bosco HTTP API (docs/API.md, docs/SERVICE-R10.md). Private to the server; Ask Bosco is its client.

    uv run --group service --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
        uvicorn bosco.service.app:app --host 127.0.0.1 --port 8420

POST /v1/decide   {version, state: {text}, questions: {id: {type: "choose", options?}}, allow_untaught?}
GET  /v1/traces/{id}
GET  /v1/models
GET  /v1/families/{id}/maps/{anatomy|wave}
"""

from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from bosco import paths
from bosco.service.core import Fleet, NotTaught

ROOT = paths.ROOT / "service"
app = FastAPI(title="Bosco", docs_url=None, redoc_url=None)
_fleet: Fleet | None = None


def fleet() -> Fleet:
    global _fleet
    if _fleet is None:
        _fleet = Fleet(ROOT, os.environ.get("BOSCO_FAMILY", "v1"))
    return _fleet


class Question(BaseModel):
    type: str = "choose"
    options: list[str] | None = None


class State(BaseModel):
    text: str


class DecideRequest(BaseModel):
    version: str
    state: State
    questions: dict[str, Question]
    allow_untaught: bool = False


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = uuid.uuid4().hex
    resp: Response = await call_next(request)
    resp.headers["x-bosco-request-id"] = rid
    return resp


@app.post("/v1/decide")
def decide(req: DecideRequest):
    f = fleet()
    try:
        spec = f.get(req.version)
    except KeyError:
        raise HTTPException(404, {"error": "unknown_version", "version": req.version}) from None
    if len(req.state.text) > 4000:
        raise HTTPException(422, {"error": "text_too_long", "field": "state.text", "max": 4000})
    t0 = time.time()
    answers, sniffs = {}, 0
    for qid, q in req.questions.items():
        if q.type != "choose":
            raise HTTPException(422, {"error": "unsupported_type", "field": f"questions.{qid}.type"})
        try:
            r = f.family.decide(spec, req.state.text, q.options, req.allow_untaught)
        except NotTaught as e:
            raise HTTPException(422, {"error": "not_taught", "field": f"questions.{qid}", "detail": str(e)}) from None
        tid = f.keep_trace(r.pop("trace"))
        sniffs += len(r["p"])
        answers[qid] = {"type": "choose", **r, "brain": {"trace_id": tid}}
    return {
        "version": spec.card["version"],
        "family": f.family.meta["id"],
        "answers": answers,
        "usage": {"sniffs": sniffs},
        "timing_ms": {"total": int(1000 * (time.time() - t0))},
    }


@app.get("/v1/traces/{tid}")
def trace(tid: str):
    t = fleet().traces.get(tid)
    if t is None:
        raise HTTPException(404, {"error": "trace_expired_or_unknown"})
    return t


@app.get("/v1/models")
def models():
    f = fleet()
    return {"family": f.family.meta, "specialists": [s.card for s in f.specs.values()], "aliases": f.latest}


@app.get("/v1/families/{fid}/maps/{view}")
def maps(fid: str, view: str):
    f = fleet()
    if fid != f.family.meta["id"] or view not in f.family.maps:
        raise HTTPException(404, {"error": "unknown_map"})
    return f.family.maps[view]
