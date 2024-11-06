import asyncio
from typing import Callable, Optional, Union

import huggingface_hub
import semchunk
import tiktoken
import tokenizers
from datasets import Dataset, concatenate_datasets, load_dataset
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
        from medrag_multi_modal.semantic_chunking import SemanticChunker


        chunker = SemanticChunker(chunk_size=256)
        chunker.chunk(
            document_dataset="geekyrakshit/grays-anatomy-test",
            chunk_dataset_repo_id="geekyrakshit/grays-anatomy-chunks-test",
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

    def chunk(
        self,
        document_dataset: Union[Dataset, str],
        chunk_dataset_repo_id: Optional[str] = None,
        overwrite_dataset: bool = False,
    ) -> Dataset:
        """
        Chunks a document dataset into smaller segments and publishes them as a new dataset.

        This function takes a document dataset, either as a HuggingFace Dataset object or a string
        representing the dataset repository ID, and chunks the documents into smaller segments using
        the specified chunker. The resulting chunks are then optionally published to a HuggingFace
        dataset repository.

        Args:
            document_dataset (Union[Dataset, str]): The document dataset to be chunked. It can be either
                a HuggingFace Dataset object or a string representing the dataset repository ID.
            chunk_dataset_repo_id (Optional[str]): The repository ID of the HuggingFace dataset to publish
                the chunks to, if provided. Defaults to None.
            overwrite_dataset (bool): Whether to overwrite the existing dataset if it exists. Defaults to False.

        Returns:
            Dataset: A HuggingFace Dataset object containing the chunks.
        """
        document_dataset = (
            load_dataset(document_dataset, split="corpus")
            if isinstance(document_dataset, str)
            else document_dataset
        ).to_list()

        chunks = []

        async def process_document(idx, document):
            document_chunks = self.chunker.chunk(str(document["text"]))
            for chunk in document_chunks:
                chunk_dict = {"document_idx": idx, "text": chunk}
                for key, value in document.items():
                    if key not in chunk_dict:
                        chunk_dict[key] = value
                chunks.append(chunk_dict)

        async def process_all_documents():
            tasks = []
            for idx, document in track(
                enumerate(document_dataset),
                total=len(document_dataset),
                description="Chunking documents",
            ):
                tasks.append(process_document(idx, document))
            await asyncio.gather(*tasks)

        asyncio.run(process_all_documents())

        chunks.sort(key=lambda x: x["document_idx"])

        dataset = Dataset.from_list(chunks)
        if chunk_dataset_repo_id:
            if huggingface_hub.repo_exists(chunk_dataset_repo_id, repo_type="dataset"):
                if not overwrite_dataset:
                    dataset = concatenate_datasets(
                        [
                            dataset,
                            load_dataset(chunk_dataset_repo_id, split="chunks"),
                        ]
                    )
            dataset.push_to_hub(repo_id=chunk_dataset_repo_id, split="chunks")

        return dataset
