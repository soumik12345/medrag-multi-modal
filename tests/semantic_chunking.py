import asyncio

from medrag_multi_modal.semantic_chunking import SemanticChunker


def test_semantic_chunking():
    chunker = SemanticChunker(chunk_size=256)
    dataset = asyncio.run(
        chunker.chunk(document_dataset="geekyrakshit/grays-anatomy-test")
    )
    assert dataset.num_rows == 120
    assert dataset.column_names == [
        "document_idx",
        "text",
        "page_idx",
        "document_name",
        "file_path",
        "file_url",
        "loader_name",
        "source",
    ]
