from medrag_multi_modal.semantic_chunking import SemanticChunker


def test_semantic_chunking():
    chunker = SemanticChunker(chunk_size=256)
    dataset = chunker.chunk(
        document_dataset="geekyrakshit/grays-anatomy-test", dataset_split="corpus"
    )
    assert dataset.num_rows == 49
    assert dataset.column_names == [
        "document_idx",
        "text",
        "page_idx",
        "document_name",
        "file_path",
        "file_url",
        "loader_name",
    ]
