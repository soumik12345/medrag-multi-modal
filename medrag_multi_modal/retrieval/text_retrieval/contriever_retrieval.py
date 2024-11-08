import json
import os
import shutil
from typing import Optional, Union

import huggingface_hub
import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
import weave
from datasets import Dataset, load_dataset
from rich.progress import track
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertPreTrainedModel,
    PreTrainedTokenizerFast,
)

from medrag_multi_modal.retrieval.common import (
    SimilarityMetric,
    argsort_scores,
    mean_pooling,
)
from medrag_multi_modal.utils import (
    fetch_from_huggingface,
    get_torch_backend,
    save_to_huggingface,
)


class ContrieverRetriever(weave.Model):
    """
    `ContrieverRetriever` is a class to perform retrieval tasks using the Contriever model.

    It provides methods to encode text data into embeddings, index a dataset of text chunks,
    and retrieve the most relevant chunks for a given query based on similarity metrics.

    Args:
        model_name (str): The name of the pre-trained model to use for encoding.
        vector_index (Optional[torch.Tensor]): The tensor containing the vector representations
            of the indexed chunks.
        chunk_dataset (Optional[list[dict]]): The weave dataset of text chunks to be indexed.
    """

    model_name: str
    _chunk_dataset: Optional[list[dict]]
    _tokenizer: PreTrainedTokenizerFast
    _model: BertPreTrainedModel
    _vector_index: Optional[torch.Tensor]

    def __init__(
        self,
        model_name: str = "facebook/contriever",
        vector_index: Optional[torch.Tensor] = None,
        chunk_dataset: Optional[list[dict]] = None,
    ):
        super().__init__(model_name=model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(get_torch_backend())
        self._vector_index = vector_index
        self._chunk_dataset = chunk_dataset

    def encode(self, corpus: list[str], batch_size: int) -> torch.Tensor:
        embeddings = []
        iterable = track(
            range(0, len(corpus), batch_size),
            description=f"Encoding corpus using {self.model_name}",
        ) if batch_size > 1 else range(0, len(corpus), batch_size)
        for idx in iterable:
            batch = corpus[idx : idx + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(get_torch_backend())
            with torch.no_grad():
                outputs = self._model(**inputs)
                batch_embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
                embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def index(
        self,
        chunk_dataset: Union[str, Dataset],
        index_repo_id: Optional[str] = None,
        cleanup: bool = True,
        batch_size: int = 32,
    ):
        """
        Indexes a dataset of text chunks and optionally saves the vector index to a file.

        This method retrieves a dataset of text chunks from a Weave reference, encodes the
        text chunks into vector representations using the Contriever model, and stores the
        resulting vector index. If an index name is provided, the vector index is saved to
        a file in the safetensors format. Additionally, if a Weave run is active, the vector
        index file is logged as an artifact to Weave.

        !!! example "Example Usage"
            ```python
            from medrag_multi_modal.retrieval.text_retrieval import ContrieverRetriever

            retriever = ContrieverRetriever()
            retriever.index(
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-contriever",
                batch_size=256,
            )
            ```

        Args:
            chunk_dataset (str): The Huggingface dataset containing the text chunks to be indexed. Either a
                dataset repository name or a dataset object can be provided.
            index_repo_id (Optional[str]): The Huggingface repository of the index artifact to be saved.
            cleanup (bool, optional): Whether to delete the local index directory after saving the vector index.
            batch_size (int, optional): The batch size to use for encoding the corpus.
        """
        self._chunk_dataset = (
            load_dataset(chunk_dataset, split="chunks")
            if isinstance(chunk_dataset, str)
            else chunk_dataset
        )
        corpus = [row["text"] for row in self._chunk_dataset]
        with torch.no_grad():
            vector_index = self.encode(corpus, batch_size)
            self._vector_index = vector_index
            if index_repo_id:
                index_save_dir = os.path.join(
                    ".huggingface", index_repo_id.split("/")[-1]
                )
                os.makedirs(index_save_dir, exist_ok=True)
                safetensors.torch.save_file(
                    {"vector_index": vector_index.cpu()},
                    os.path.join(index_save_dir, "vector_index.safetensors"),
                )
                commit_type = (
                    "update"
                    if huggingface_hub.repo_exists(index_repo_id, repo_type="model")
                    else "add"
                )
                with open(
                    os.path.join(index_save_dir, "config.json"), "w"
                ) as config_file:
                    json.dump(
                        {"model_name": self.model_name},
                        config_file,
                        indent=4,
                    )
                save_to_huggingface(
                    index_repo_id,
                    index_save_dir,
                    commit_message=f"{commit_type}: Contriever index",
                )
                if cleanup:
                    shutil.rmtree(index_save_dir)

    @classmethod
    def from_index(cls, chunk_dataset: Union[str, Dataset], index_repo_id: str):
        """
        Creates an instance of the class from a Weave artifact.

        This method retrieves a vector index and metadata from a Weave artifact stored in
        Weights & Biases (wandb). It also retrieves a dataset of text chunks from a Weave
        reference. The vector index is loaded from a safetensors file and moved to the
        appropriate device (CPU or GPU). The text chunks are converted into a list of
        dictionaries. The method then returns an instance of the class initialized with
        the retrieved model name, vector index, and chunk dataset.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            from medrag_multi_modal.retrieval.text_retrieval import ContrieverRetriever

            load_dotenv()
            retriever = ContrieverRetriever().from_index(
                index_repo_id="geekyrakshit/grays-anatomy-index-contriever",
                chunk_dataset="geekyrakshit/grays-anatomy-chunks-test",
            )
            ```

        Args:
            chunk_dataset (str): The Huggingface dataset containing the text chunks to be indexed. Either a
                dataset repository name or a dataset object can be provided.
            index_repo_id (Optional[str]): The Huggingface repository of the index artifact to be saved.

        Returns:
            An instance of the class initialized with the retrieved model name, vector index,
            and chunk dataset.
        """
        index_dir = fetch_from_huggingface(index_repo_id, ".huggingface")
        with safetensors.torch.safe_open(
            os.path.join(index_dir, "vector_index.safetensors"), framework="pt"
        ) as f:
            vector_index = f.get_tensor("vector_index")
        device = torch.device(get_torch_backend())
        vector_index = vector_index.to(device)
        chunk_dataset = (
            load_dataset(chunk_dataset, split="chunks")
            if isinstance(chunk_dataset, str)
            else chunk_dataset
        )
        with open(os.path.join(index_dir, "config.json"), "r") as config_file:
            metadata = json.load(config_file)
        return cls(
            model_name=metadata["model_name"],
            vector_index=vector_index,
            chunk_dataset=chunk_dataset,
        )

    @weave.op()
    def retrieve(
        self,
        query: str,
        top_k: int = 2,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        """
        Retrieves the top-k most relevant chunks for a given query using the specified similarity metric.

        This method encodes the input query into an embedding and computes similarity scores between
        the query embedding and the precomputed vector index. The similarity metric can be either
        cosine similarity or Euclidean distance. The top-k chunks with the highest similarity scores
        are returned as a list of dictionaries, each containing a chunk and its corresponding score.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            from medrag_multi_modal.retrieval.text_retrieval import ContrieverRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = ContrieverRetriever().from_index(
                index_repo_id="geekyrakshit/grays-anatomy-index-contriever",
                chunk_dataset="geekyrakshit/grays-anatomy-chunks-test",
            )
            retrieved_chunks = retriever.retrieve(query="What are Ribosomes?")
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 2.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        query = [query]
        device = torch.device(get_torch_backend())
        with torch.no_grad():
            query_embedding = self.encode(query, batch_size=1).to(device)
            if metric == SimilarityMetric.EUCLIDEAN:
                scores = torch.squeeze(query_embedding @ self._vector_index.T)
            else:
                scores = F.cosine_similarity(query_embedding, self._vector_index)
            scores = scores.cpu().numpy().tolist()
        scores = argsort_scores(scores, descending=True)[:top_k]
        retrieved_chunks = []
        for score in scores:
            retrieved_chunks.append(
                {
                    **self._chunk_dataset[score["original_index"]],
                    **{"score": score["item"]},
                }
            )
        return retrieved_chunks

    @weave.op()
    def predict(
        self,
        query: str,
        top_k: int = 2,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        """
        Predicts the top-k most relevant chunks for a given query using the specified similarity metric.

        This function is a wrapper around the `retrieve` method. It takes an input query string,
        retrieves the top-k most relevant chunks from the precomputed vector index based on the
        specified similarity metric, and returns the results as a list of dictionaries, each containing
        a chunk and its corresponding relevance score.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            from medrag_multi_modal.retrieval.text_retrieval import ContrieverRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = ContrieverRetriever().from_index(
                index_repo_id="geekyrakshit/grays-anatomy-index-contriever",
                chunk_dataset="geekyrakshit/grays-anatomy-chunks-test",
            )
            retrieved_chunks = retriever.predict(query="What are Ribosomes?")
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 2.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring. Defaults to cosine similarity.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        return self.retrieve(query, top_k, metric)
