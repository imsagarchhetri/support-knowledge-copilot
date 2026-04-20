from app.generation import AnswerService


def test_grounded_answer_has_verified_citation():
    ans = AnswerService().ask("How do I reset my password?")
    assert not ans.refused
    assert ans.citations, "expected at least one verified citation"
    assert all(c.supported for c in ans.citations)
    assert ans.confidence.overall > 0.5


def test_refuses_out_of_corpus_question():
    ans = AnswerService().ask("What is your company's stock price today?")
    assert ans.refused
    assert "could not find" in ans.answer.lower()


def test_answer_only_cites_retrieved_chunks():
    ans = AnswerService().ask("What does error code BIL-402 mean?")
    retrieved_ids = {h.chunk_id for h in ans.retrieved}
    for c in ans.citations:
        assert c.chunk_id in retrieved_ids
