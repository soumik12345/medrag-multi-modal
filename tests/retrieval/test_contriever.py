from medrag_multi_modal.retrieval.text_retrieval import ContrieverRetriever


def test_contriever_retriever():
    retriever = ContrieverRetriever().from_index(
        index_repo_id="geekyrakshit/grays-anatomy-index-contriever",
        chunk_dataset="geekyrakshit/grays-anatomy-chunks-test",
    )
    retrieved_chunks = retriever.retrieve(query="What are Ribosomes?")
    assert len(retrieved_chunks) == 2
    for chunk in retrieved_chunks:
        assert "score" in chunk
        assert "text" in chunk
        assert chunk["score"] > 0
        assert "ribosomes" in chunk["text"].lower()
