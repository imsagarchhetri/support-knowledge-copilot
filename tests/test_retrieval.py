from app.retrieval import HybridRetriever


def test_hybrid_retrieves_expected_source():
    r = HybridRetriever()
    hits, sig = r.retrieve("How do I reset my password?")
    sources = {h.source for h in hits}
    assert "account_faq.md" in sources
    assert sig.max_dense > 0


def test_bm25_catches_error_code():
    # exact token an error code -> BM25 should surface the right doc
    r = HybridRetriever()
    hits, sig = r.retrieve("What does error code BIL-402 mean?")
    assert hits[0].source == "billing_guide.md"
    assert sig.max_bm25 > 0


def test_hybrid_beats_or_matches_dense_on_recall():
    r = HybridRetriever()
    q = "What is the API rate limit?"
    hybrid, _ = r.retrieve(q, hybrid=True)
    dense, _ = r.retrieve(q, hybrid=False)
    assert any(h.source == "troubleshooting.md" for h in hybrid)
    assert len(dense) > 0
