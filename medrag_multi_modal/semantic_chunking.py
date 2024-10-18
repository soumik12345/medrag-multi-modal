from typing import Callable, Optional, Union

import semchunk
import tiktoken
import tokenizers
import weave
from rich.progress import track
from transformers import PreTrainedTokenizer

TOKENIZER_OR_TOKEN_COUNTER = Union[
    str,
    tiktoken.Encoding,
    PreTrainedTokenizer,
    tokenizers.Tokenizer,
    Callable[[str], int],
]


class SemanticChunker:
    """
    SemanticChunker is a class that chunks documents into smaller segments and
    publishes them as datasets.

    This class uses the `semchunk` library to break down large documents into
    smaller, manageable chunks based on a specified tokenizer or token counter.
    This is particularly useful for processing large text datasets where
    smaller segments are needed for analysis or other operations.

    !!! example "Example Usage"
        ```python
        import weave
        from dotenv import load_dotenv

        from medrag_multi_modal.semantic_chunking import SemanticChunker

        load_dotenv()
        weave.init(project_name="ml-colabs/medrag-multi-modal")
        chunker = SemanticChunker(chunk_size=256)
        chunker.chunk_and_publish(
            document_dataset_name="grays-anatomy-text:v13",
            chunk_dataset_name="grays-anatomy-chunks",
        )
        ```

    Args:
        tokenizer_or_token_counter (TOKENIZER_OR_TOKEN_COUNTER): The tokenizer or
            token counter to be used for chunking.
        chunk_size (Optional[int]): The size of each chunk. If not specified, the
            default chunk size from `semchunk` will be used.
        max_token_chars (Optional[int]): The maximum number of characters per token.
            If not specified, the default value from `semchunk` will be used.
        memoize (bool): Whether to memoize the chunking process for efficiency.
            Default is True.
    """

    def __init__(
        self,
        tokenizer_or_token_counter: TOKENIZER_OR_TOKEN_COUNTER = "o200k_base",
        chunk_size: Optional[int] = None,
        max_token_chars: Optional[int] = None,
        memoize: bool = True,
    ) -> None:
        self.chunker = semchunk.chunkerify(
            tokenizer_or_token_counter,
            chunk_size=chunk_size,
            max_token_chars=max_token_chars,
            memoize=memoize,
        )

    def chunk_and_publish(
        self, document_dataset_name: str, chunk_dataset_name: Optional[str] = None
    ) -> None:
        document_dataset = weave.ref(document_dataset_name).get().rows
        chunks = []
        for idx, document in track(
            enumerate(document_dataset), description="Chunking documents"
        ):
            document_chunks = self.chunker.chunk(str(document["text"]))
            for chunk in document_chunks:
                chunks.append(
                    {
                        "document_idx": idx,
                        "document_name": document["document_name"],
                        "page_idx": document["page_idx"],
                        "text": chunk,
                    }
                )
        weave.publish(weave.Dataset(name=chunk_dataset_name, rows=chunks))
