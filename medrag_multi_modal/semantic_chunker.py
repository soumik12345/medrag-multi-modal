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
