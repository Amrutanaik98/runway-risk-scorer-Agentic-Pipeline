"""
extract/extract_signals.py  —  Sprint 3: LLM extraction (LangChain + Groq, free tier).

The agent (Sprint 2) gathers RAW signals — a headline tagged 'news_mention'.
This step reads each headline with an LLM (Llama, served free by Groq) and
EXTRACTS structure: signal type, any amount, any date, a confidence score.
Vague mention -> precise typed signal.

Why Groq: it's a real API (so this stays Sprint 3's "API model"), it serves Llama
models, and it has a free tier — no cost. Sprint 4 ("local Llama via Ollama") then
becomes a clean comparison: same model family, API vs. local.

The boundary that keeps the design honest:
- The LLM EXTRACTS structure. It does NOT decide risk, and it does NOT validate.
- Every extracted signal stays validated_by = None (P2). A human validates before
  it counts. The LLM reads and structures; the human judges.
- If unsure, the model says so via low confidence rather than inventing a value.

Setup:
    pip install langchain langchain-groq pydantic
    Get a FREE key at https://console.groq.com  (API Keys -> Create)
    PowerShell:  $env:GROQ_API_KEY = "gsk_..."
Run:
    python -m extract.extract_signals data/raw/news_signals.json --limit 2
    python -m extract.extract_signals data/raw/news_signals.json --out data/extracted/signals.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field  # noqa: E402


class ExtractedSignal(BaseModel):
    """Structured fields the LLM must return (LangChain forces typed output)."""
    signal_type: str = Field(
        description="one of: funding_round, funding_stage, layoff, executive_change, "
                    "security_issue, product_launch, news_mention"
    )
    amount_usd: Optional[str] = Field(
        default=None, description="funding amount if mentioned, e.g. '$2B'; else null"
    )
    occurred_date: Optional[str] = Field(
        default=None, description="ISO date YYYY-MM-DD if clearly present; else null"
    )
    confidence: int = Field(
        description="0-100 confidence. Low if vague. Never invent details."
    )
    reasoning: str = Field(description="one short sentence: why this type/these values")


def build_extractor():
    from langchain_groq import ChatGroq
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            'GROQ_API_KEY not set. Get a free key at https://console.groq.com, then:\n'
            '  PowerShell:  $env:GROQ_API_KEY = "gsk_..."'
        )
    # free, fast Llama model served by Groq
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    return llm.with_structured_output(ExtractedSignal)


SYSTEM = (
    "You extract structured financial/company signals from a news headline. "
    "Return only what the text supports. If the headline is a vague mention with no "
    "funding/layoff/exec/security/launch content, use signal_type 'news_mention' and "
    "low confidence. Never invent an amount or a date that isn't in the text."
)


def extract_one(extractor, raw: dict) -> dict:
    headline = raw.get("signal_title", "")
    company = raw.get("company_id", "")
    prompt = f"{SYSTEM}\n\nCompany: {company}\nHeadline: {headline}\n\nExtract the signal."
    result: ExtractedSignal = extractor.invoke(prompt)

    enriched = dict(raw)
    enriched["signal_type"] = result.signal_type
    if result.amount_usd:
        enriched["signal_value"] = result.amount_usd
    if result.occurred_date:
        enriched["occurred_date"] = result.occurred_date
    enriched["score"] = result.confidence
    enriched["extraction_reasoning"] = result.reasoning
    enriched["extraction_model"] = "groq:openai/gpt-oss-20b"
    enriched["validated_by"] = None  # P2: STILL needs a human
    return enriched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    raw_signals = json.load(open(args.raw_path))
    if args.limit:
        raw_signals = raw_signals[: args.limit]
    if not raw_signals:
        print("No raw signals to extract. Run the agent/fetch first.")
        return 0

    extractor = build_extractor()
    print("LLM EXTRACTION (LangChain + Groq/Llama, free tier)")
    print("=" * 52)
    out = []
    for raw in raw_signals:
        enriched = extract_one(extractor, raw)
        out.append(enriched)
        print(f"\n  {enriched['company_id']}: {raw.get('signal_title','')[:60]}")
        print(f"    -> type={enriched['signal_type']}  value={enriched.get('signal_value')}  "
              f"date={enriched.get('occurred_date')}  conf={enriched['score']}")
        print(f"    reasoning: {enriched['extraction_reasoning']}")
        print(f"    validated_by: None  (still UNVALIDATED \u2014 a human must confirm, P2)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\n[saved] {len(out)} extracted signals -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())