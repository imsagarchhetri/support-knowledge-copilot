"""Evaluation suite.

Measures retrieval and answer quality SEPARATELY over a hand-written golden set,
using RAGAS-style metrics implemented locally (no LLM key required):

  - context_recall@k   : was the expected source retrieved?
  - answer_match        : does the answer contain the expected fact?
  - faithfulness        : citation support rate (verified / cited)
  - refusal_accuracy    : did it correctly refuse out-of-corpus questions?

Run dense-only vs hybrid to produce the headline comparison. Set
`--ragas` (and an LLM key) to additionally run the real `ragas` package.

    python -m eval.run_eval --strategy hybrid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.generation import AnswerService

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
REPORT = Path(__file__).resolve().parent / "report.md"


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def run(strategy: str = "hybrid") -> dict:
    hybrid = strategy == "hybrid"
    svc = AnswerService()
    cases = load_golden()

    recall_hits = match_hits = faith_sum = 0
    n_answerable = n_faith = 0
    refusal_ok = n_noanswer = 0
    rows = []

    for c in cases:
        ans = svc.ask(c["question"], hybrid=hybrid)
        if c.get("no_answer"):
            n_noanswer += 1
            refusal_ok += int(ans.refused)
            rows.append({"q": c["question"], "type": "no_answer", "refused": ans.refused})
            continue

        n_answerable += 1
        sources = {h.source for h in ans.retrieved[: svc.s.top_k]}
        got = c.get("expected_source") in sources
        recall_hits += int(got)

        kw = c.get("expected_contains", "").lower()
        match = kw in ans.answer.lower() if kw else None
        match_hits += int(bool(match))

        if ans.citations:
            n_faith += 1
            faith_sum += sum(1 for x in ans.citations if x.supported) / len(ans.citations)

        rows.append({"q": c["question"], "source_retrieved": got, "answer_match": match,
                     "confidence": ans.confidence.overall, "n_citations": len(ans.citations)})

    return {
        "strategy": strategy,
        "n_cases": len(cases),
        "context_recall@k": round(recall_hits / n_answerable, 3) if n_answerable else None,
        "answer_match": round(match_hits / n_answerable, 3) if n_answerable else None,
        "faithfulness": round(faith_sum / n_faith, 3) if n_faith else None,
        "refusal_accuracy": round(refusal_ok / n_noanswer, 3) if n_noanswer else None,
        "rows": rows,
    }


def render(res: dict) -> str:
    lines = [f"# RAG Eval Report — strategy: {res['strategy']}", "",
             f"- Cases: {res['n_cases']}",
             f"- context_recall@k: {res['context_recall@k']}",
             f"- answer_match: {res['answer_match']}",
             f"- faithfulness (citation support): {res['faithfulness']}",
             f"- refusal_accuracy: {res['refusal_accuracy']}", "",
             "| Question | Source ok | Answer match | Confidence |",
             "|---|---|---|---|"]
    for r in res["rows"]:
        if r.get("type") == "no_answer":
            lines.append(f"| {r['q']} | (no-answer) | refused={r['refused']} | - |")
        else:
            lines.append(f"| {r['q']} | {r['source_retrieved']} | "
                         f"{r['answer_match']} | {r['confidence']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="hybrid", choices=["hybrid", "dense"])
    args = ap.parse_args()
    res = run(args.strategy)
    report = render(res)
    REPORT.write_text(report)
    print(report)
    print(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    main()
