from types import SimpleNamespace

import pytest

from src import query


def test_format_docs_includes_metadata():
    docs = [
        SimpleNamespace(page_content="Answer one", metadata={"source": "paper1.pdf", "page": 2}),
        SimpleNamespace(page_content="Answer two", metadata={"source": "paper2.pdf", "page": 5}),
    ]

    formatted = query.format_docs(docs)

    assert "[Document 0]" in formatted
    assert "paper1.pdf" in formatted
    assert "Answer two" in formatted


def test_create_prompt_mentions_question_and_history():
    history = [
        ("Q1", "A1"),
        ("Q2", "A2"),
        ("Q3", "A3"),
        ("Q4", "A4"),
    ]

    prompt = query.create_prompt("Context text", "What is up?", history)
    assert "Context text" in prompt
    assert "What is up?" in prompt
    assert "Previous Q: Q2" not in prompt  # trimmed to last 3 exchanges
    assert "Previous Q: Q4" in prompt
    assert "Previous A: A3" in prompt


def test_load_vecstore_uses_custom_directory(monkeypatch):
    class DummyEmbeddings:
        def __init__(self, *_, **__):
            pass

    class DummyCollection:
        def __init__(self):
            self._count = 3

        def count(self):
            return self._count

    class DummyChroma:
        def __init__(self, persist_directory, embedding_function):
            self.persist_directory = persist_directory
            self.embedding_function = embedding_function
            self._collection = DummyCollection()

    monkeypatch.setattr(query, "HuggingFaceEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(query, "Chroma", DummyChroma)

    store = query.load_vecstore("/tmp/vector")
    assert store.persist_directory == "/tmp/vector"
    assert isinstance(store._collection, DummyCollection)


def test_init_llm_uses_llama32(monkeypatch):
    captured = {}

    class DummyLLM:
        def __init__(self, *_, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(query, "OllamaLLM", DummyLLM)

    llm = query.init_llm()
    assert captured["model"] == "llama3.2"
    assert captured["temperature"] == 0.1
    assert isinstance(llm, DummyLLM)


def test_query_returns_answer_and_sources(monkeypatch, capsys):
    docs = [
        SimpleNamespace(page_content="Chunk content", metadata={"source": "doc.txt", "page": 1})
    ]

    class DummyVectorstore:
        def __init__(self):
            self.requested = None

        def similarity_search(self, question, k):
            self.requested = (question, k)
            return docs

    class DummyLLM:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return "Final answer"

    store = DummyVectorstore()
    llm = DummyLLM()

    history = []
    result = query.query(store, llm, "Sample question", history=history, k=1)

    captured = capsys.readouterr()
    assert "Sample question" in captured.out
    assert store.requested == ("Sample question", 3)
    assert "Final answer" in captured.out
    assert result["answer"] == "Final answer"
    assert result["source_documents"] == docs
