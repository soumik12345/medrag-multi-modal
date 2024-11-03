from medrag_multi_modal.retrieval.text_retrieval import BM25sRetriever


def test_bm25s_retriever():
    retriever = BM25sRetriever().from_index(
        index_repo_id="geekyrakshit/grays-anatomy-index"
    )
    retrieved_chunks = retriever.predict(query="What are Ribosomes?", top_k=2)
    assert len(retrieved_chunks) == 2
    for chunk in retrieved_chunks:
        assert "score" in chunk
        assert "text" in chunk
        assert chunk["score"] > 0
        assert "ribosomes" in chunk["text"].lower()
