import os
from glob import glob
from typing import Optional

import bm25s
import weave
from Stemmer import Stemmer

import wandb

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

    language: str
    use_stemmer: bool
    _retriever: Optional[bm25s.BM25]

    def __init__(
        self,
        language: str = "english",
        use_stemmer: bool = True,
        retriever: Optional[bm25s.BM25] = None,
    ):
        super().__init__(language=language, use_stemmer=use_stemmer)
        self._retriever = retriever or bm25s.BM25()

    def index(self, chunk_dataset_name: str, index_name: Optional[str] = None):
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

            import wandb
            from medrag_multi_modal.retrieval import BM25sRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            wandb.init(project="medrag-multi-modal", entity="ml-colabs", job_type="bm25s-index")
            retriever = BM25sRetriever()
            retriever.index(chunk_dataset_name="grays-anatomy-text:v13", index_name="grays-anatomy-bm25s")
            ```

        Args:
            chunk_dataset_name (str): The name of the dataset containing text chunks to be indexed.
            index_name (Optional[str]): The name to save the index under. If provided, the index
                is saved to disk and logged as a Weights & Biases artifact.
        """
        chunk_dataset = weave.ref(chunk_dataset_name).get().rows
        corpus = [row["text"] for row in chunk_dataset]
        corpus_tokens = bm25s.tokenize(
            corpus,
            stopwords=LANGUAGE_DICT[self.language],
            stemmer=Stemmer(self.language) if self.use_stemmer else None,
        )
        self._retriever.index(corpus_tokens)
        if index_name:
            self._retriever.save(
                index_name, corpus=[dict(row) for row in chunk_dataset]
            )
            if wandb.run:
                artifact = wandb.Artifact(
                    name=index_name,
                    type="bm25s-index",
                    metadata={
                        "language": self.language,
                        "use_stemmer": self.use_stemmer,
                    },
                )
                artifact.add_dir(index_name, name=index_name)
                artifact.save()

    @classmethod
    def from_wandb_artifact(cls, index_artifact_address: str):
        """
        Creates an instance of the class from a Weights & Biases artifact.

        This class method retrieves a BM25 index artifact from Weights & Biases,
        downloads the artifact, and loads the BM25 retriever with the index and its
        associated corpus. The method also extracts metadata from the artifact to
        initialize the class instance with the appropriate language and stemming
        settings.

        Args:
            index_artifact_address (str): The address of the Weights & Biases artifact
                containing the BM25 index.

        Returns:
            An instance of the class initialized with the BM25 retriever and metadata
            from the artifact.
        """
        if wandb.run:
            artifact = wandb.run.use_artifact(
                index_artifact_address, type="bm25s-index"
            )
            artifact_dir = artifact.download()
        else:
            api = wandb.Api()
            artifact = api.artifact(index_artifact_address)
            artifact_dir = artifact.download()
        index_name = glob(os.path.join(artifact_dir, "*"))[0].split("/")[-1]
        retriever = bm25s.BM25.load(index_name, load_corpus=True)
        metadata = artifact.metadata
        return cls(
            language=metadata["language"],
            use_stemmer=metadata["use_stemmer"],
            retriever=retriever,
        )

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

            from medrag_multi_modal.retrieval import BM25sRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = BM25sRetriever.from_wandb_artifact(
                index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-bm25s:v2"
            )
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
            retrieved_chunks.append({"chunk": chunk, "score": score})
        return retrieved_chunks
