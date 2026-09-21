# Agentic Runway-Risk Pipeline — Week 1: Foundation & Schema

The typed-data foundation the rest of the pipeline sits on. No agent or LLM yet —
Week 1 is about getting real signals in and rejecting bad data at the door.

## What's here
- `schema/signal.py` — the pydantic `Signal` model. Every signal must conform:
  typed fields, a real source_url, a non-future date, score 0-100, known signal_type.
- `ingest/load_signals.py` — loads a JSON file, validates each record against the
  schema, and reports what passed and what was rejected (with reasons).
- `data/samples/sample_signals.json` — test fixture: 3 valid signals + 4 that are
  deliberately broken (bad type, bad URL, future date, out-of-range score).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run it
```bash
python -m ingest.load_signals data/samples/sample_signals.json
```
Expected: 3 valid, 4 rejected — each rejection naming its reason.

## Week 1 done when
- The schema validates the sample set (3 pass, 4 rejected with reasons).
- You can add a new signal to the JSON and watch it either pass or get caught.

## Next (Week 2)
Build the LangGraph research agent that will *gather* real signals to feed this
validator — the agentic front end. This foundation stays unchanged underneath it.
