import json
import os
import shutil
from typing import Optional, Union

import bm25s
import weave
from datasets import Dataset, load_dataset
from Stemmer import Stemmer

from medrag_multi_modal.utils import (
    fetch_from_huggingface,
    is_existing_huggingface_repo,
    save_to_huggingface,
)

LANGUAGE_DICT = {
    "english": "en",
    "french": "fr",
    "german": "de",
}


class BM25sRetriever(weave.Model):
    """
    `BM25sRetriever` is a class that provides functionality for indexing and
    retrieving documents using the [BM25-Sparse](https://github.com/xhluca/bm25s).

    Args:
        language (str): The language of the documents to be indexed and retrieved.
        use_stemmer (bool): A flag indicating whether to use stemming during tokenization.
        retriever (Optional[bm25s.BM25]): An instance of the BM25 retriever. If not provided,
            a new instance is created.
    """

    language: Optional[str]
    use_stemmer: bool = True
    _retriever: Optional[bm25s.BM25]

    def __init__(
        self,
        language: str = "english",
        use_stemmer: bool = True,
        retriever: Optional[bm25s.BM25] = None,
    ):
        super().__init__(language=language, use_stemmer=use_stemmer)
        self._retriever = retriever or bm25s.BM25()

    def index(
        self,
        chunk_dataset: Union[Dataset, str],
        index_repo_id: Optional[str] = None,
        cleanup: bool = True,
    ):
        """
        Indexes a dataset of text chunks using the BM25 algorithm.

        This function takes a dataset of text chunks identified by `chunk_dataset_name`,
        tokenizes the text using the BM25 tokenizer with optional stemming, and indexes
        the tokenized text using the BM25 retriever. If an `index_name` is provided, the
        index is saved to disk and logged as a Weights & Biases artifact.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            from medrag_multi_modal.retrieval.text_retrieval import BM25sRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = BM25sRetriever()
            retriever.index(
                chunk_dataset="geekyrakshit/grays-anatomy-chunks-test",
                index_repo_id="geekyrakshit/grays-anatomy-index",
            )
            ```

        Args:
            chunk_dataset_name (str): The name of the dataset containing text chunks to be indexed.
            index_name (Optional[str]): The name to save the index under. If provided, the index
                is saved to disk and logged as a Weights & Biases artifact.
        """
        chunk_dataset = (
            load_dataset(chunk_dataset, split="chunks")
            if isinstance(chunk_dataset, str)
            else chunk_dataset
        )
        corpus = [row["text"] for row in chunk_dataset]
        corpus_tokens = bm25s.tokenize(
            corpus,
            stopwords=LANGUAGE_DICT[self.language],
            stemmer=Stemmer(self.language) if self.use_stemmer else None,
        )
        self._retriever.index(corpus_tokens)
        if index_repo_id:
            os.makedirs(".huggingface", exist_ok=True)
            index_save_dir = os.path.join(".huggingface", index_repo_id.split("/")[-1])
            self._retriever.save(
                index_save_dir, corpus=[dict(row) for row in chunk_dataset]
            )
            commit_type = "update" if is_existing_huggingface_repo(index_repo_id) else "add"
            with open(os.path.join(index_save_dir, "config.json"), "w") as config_file:
                json.dump(
                    {
                        "language": self.language,
                        "use_stemmer": self.use_stemmer,
                    },
                    config_file,
                    indent=4,
                )
            save_to_huggingface(
                index_repo_id,
                index_save_dir,
                commit_message=f"{commit_type}: BM25s index",
            )
            if cleanup:
                shutil.rmtree(index_save_dir)

    @classmethod
    def from_index(cls, index_repo_id: str):
        """
        Creates an instance of the class from a Weights & Biases artifact.

        This class method retrieves a BM25 index artifact from Weights & Biases,
        downloads the artifact, and loads the BM25 retriever with the index and its
        associated corpus. The method also extracts metadata from the artifact to
        initialize the class instance with the appropriate language and stemming
        settings.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            from medrag_multi_modal.retrieval.text_retrieval import BM25sRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = BM25sRetriever()
            retriever = BM25sRetriever().from_index(index_repo_id="geekyrakshit/grays-anatomy-index")
            ```

        Args:
            index_artifact_address (str): The address of the Weights & Biases artifact
                containing the BM25 index.

        Returns:
            An instance of the class initialized with the BM25 retriever and metadata
            from the artifact.
        """
        index_dir = fetch_from_huggingface(index_repo_id, ".huggingface")
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        with open(os.path.join(index_dir, "config.json"), "r") as config_file:
            config = json.load(config_file)
        return cls(retriever=retriever, **config)

    @weave.op()
    def retrieve(self, query: str, top_k: int = 2):
        """
        Retrieves the top-k most relevant chunks for a given query using the BM25 algorithm.

        This method tokenizes the input query using the BM25 tokenizer, which takes into
        account the language-specific stopwords and optional stemming. It then retrieves
        the top-k most relevant chunks from the BM25 index based on the tokenized query.
        The results are returned as a list of dictionaries, each containing a chunk and
        its corresponding relevance score.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            from medrag_multi_modal.retrieval.text_retrieval import BM25sRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = BM25sRetriever()
            retriever = BM25sRetriever().from_index(index_repo_id="geekyrakshit/grays-anatomy-index")
            retrieved_chunks = retriever.retrieve(query="What are Ribosomes?")
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 2.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its
                relevance score.
        """
        query_tokens = bm25s.tokenize(
            query,
            stopwords=LANGUAGE_DICT[self.language],
            stemmer=Stemmer(self.language) if self.use_stemmer else None,
        )
        results = self._retriever.retrieve(query_tokens, k=top_k)
        retrieved_chunks = []
        for chunk, score in zip(
            results.documents.flatten().tolist(),
            results.scores.flatten().tolist(),
        ):
            retrieved_chunks.append({**chunk, **{"score": score}})
        return retrieved_chunks

    @weave.op()
    def predict(self, query: str, top_k: int = 2):
        """
        Predicts the top-k most relevant chunks for a given query using the BM25 algorithm.

        This function is a wrapper around the `retrieve` method. It takes an input query string,
        tokenizes it using the BM25 tokenizer, and retrieves the top-k most relevant chunks from
        the BM25 index. The results are returned as a list of dictionaries, each containing a chunk
        and its corresponding relevance score.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            from medrag_multi_modal.retrieval.text_retrieval import BM25sRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = BM25sRetriever()
            retriever = BM25sRetriever().from_index(index_repo_id="geekyrakshit/grays-anatomy-index")
            retrieved_chunks = retriever.predict(query="What are Ribosomes?")
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 2.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        return self.retrieve(query, top_k)
