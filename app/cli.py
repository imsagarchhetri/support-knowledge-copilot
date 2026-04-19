"""Command-line interface.

    python -m app.cli ingest [--strategy heading|fixed]
    python -m app.cli ask "How do I reset my password?" [--dense-only]
"""
from __future__ import annotations

import argparse
import json

from .generation import AnswerService
from .index_service import build_index


def main():
    ap = argparse.ArgumentParser(description="Support Knowledge Copilot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--strategy", default=None, choices=["heading", "fixed"])

    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--dense-only", action="store_true")
    a.add_argument("--top-k", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "ingest":
        n = build_index(strategy=args.strategy)
        print(f"Indexed {n} chunks.")
    elif args.cmd == "ask":
        ans = AnswerService().ask(args.question, hybrid=not args.dense_only,
                                  top_k=args.top_k)
        print(json.dumps(ans.model_dump(), indent=2))


if __name__ == "__main__":
    main()
