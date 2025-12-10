import os
from types import SimpleNamespace

import pytest

from src import ingest


def test_load_documents_reads_text_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample = data_dir / "sample.txt"
    sample.write_text("Hello world!", encoding="utf-8")

    docs = ingest.load_documents(str(data_dir))

    assert len(docs) == 1
    assert "Hello world!" in docs[0].page_content


def test_chunk_documents_splits_large_text(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample = data_dir / "long.txt"
    sample.write_text("Lorem ipsum " * 200, encoding="utf-8")

    docs = ingest.load_documents(str(data_dir))
    chunks = ingest.chunk_documents(docs)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)


def test_create_vecstore_uses_persist_dir(monkeypatch, tmp_path):
    dummy_chunks = [SimpleNamespace(page_content="chunk 1"), SimpleNamespace(page_content="chunk 2")]
    created = {}

    class DummyEmbeddings:
        def __init__(self, *args, **kwargs):
            created["embeddings_kwargs"] = kwargs

    class DummyChroma:
        def __init__(self, documents, embedding, persist_directory):
            self.documents = documents
            self.embedding = embedding
            self.persist_directory = persist_directory

        @classmethod
        def from_documents(cls, documents, embedding, persist_directory):
            return cls(documents, embedding, persist_directory)

    monkeypatch.setattr(ingest, "HuggingFaceEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(ingest, "Chroma", DummyChroma)

    persist_dir = tmp_path / "vector"
    vectorstore = ingest.create_vecstore(dummy_chunks, str(persist_dir))

    assert vectorstore.documents == dummy_chunks
    assert vectorstore.persist_directory == str(persist_dir)
    assert created["embeddings_kwargs"]["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
