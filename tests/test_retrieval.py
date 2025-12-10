from types import SimpleNamespace

from src import query


def test_similarity_search_outputs_snippets(monkeypatch, capsys):
    docs = [
        SimpleNamespace(page_content="Machine learning snippet", metadata={"source": "paper1.pdf", "page": 1}),
        SimpleNamespace(page_content="Deep learning snippet", metadata={"source": "paper2.pdf", "page": 3}),
    ]

    class DummyVectorstore:
        def similarity_search(self, question, k):
            return docs

    class DummyLLM:
        def invoke(self, prompt):
            return "stub answer"

    result = query.query(DummyVectorstore(), DummyLLM(), "machine learning overview", history=[])

    output = capsys.readouterr().out
    assert "machine learning overview" in output
    assert "Machine learning snippet" in output
    assert "Sources" in output
    assert result["source_documents"] == docs
