"""The UML-spec RAG must load its PDF: nothing used to, so every container's
vector store stayed empty and "according to the UML spec" answers had no
context. Indexing runs once in the background and survives an interrupted run.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

import agent_setup


class _FakeStore:
    """The LangChain VectorStore calls the indexer makes (Chroma's get())."""

    def __init__(self, documents=()):
        self.documents = list(documents)

    def get(self):
        return {"ids": [str(i) for i in range(len(self.documents))]}

    def delete(self, ids):
        self.documents = [d for i, d in enumerate(self.documents) if str(i) not in ids]

    def add_documents(self, documents):
        self.documents.extend(documents)


class _FakeRAG:
    def __init__(self, documents=()):
        self.vector_store = _FakeStore(documents)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def _point_marker_at(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_setup, "_UML_INDEX_DONE", tmp_path / "store" / ".indexed")


def test_an_empty_store_gets_the_uml_specification(tmp_path, monkeypatch):
    _point_marker_at(tmp_path, monkeypatch)
    rag = _FakeRAG()

    agent_setup._index_uml_specs_in_background(rag).join(timeout=60)

    chunks = rag.vector_store.documents
    assert len(chunks) > 2000  # the 796-page UML 2.5.1 specification
    assert any("Association" in chunk.page_content for chunk in chunks)
    assert agent_setup._UML_INDEX_DONE.exists()


def test_a_finished_index_is_not_rebuilt(tmp_path, monkeypatch):
    _point_marker_at(tmp_path, monkeypatch)
    agent_setup._UML_INDEX_DONE.parent.mkdir()
    agent_setup._UML_INDEX_DONE.touch()
    rag = _FakeRAG(documents=["kept"])

    assert agent_setup._index_uml_specs_in_background(rag) is None
    assert rag.vector_store.documents == ["kept"]


def test_a_partial_index_is_cleared_and_redone(tmp_path, monkeypatch):
    _point_marker_at(tmp_path, monkeypatch)
    monkeypatch.setattr(agent_setup, "_read_uml_specs", lambda: [])
    rag = _FakeRAG(documents=["partial"])  # interrupted run: chunks but no marker

    agent_setup._index_uml_specs_in_background(rag).join(timeout=5)

    assert rag.vector_store.documents == []
    assert agent_setup._UML_INDEX_DONE.exists()


def test_a_failed_index_is_retried_next_start(tmp_path, monkeypatch):
    _point_marker_at(tmp_path, monkeypatch)

    def no_key(documents):
        raise RuntimeError("embedding failed")

    rag = _FakeRAG()
    rag.vector_store.add_documents = no_key

    agent_setup._index_uml_specs_in_background(rag).join(timeout=60)

    assert not agent_setup._UML_INDEX_DONE.exists()
