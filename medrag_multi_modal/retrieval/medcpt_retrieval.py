import os
from typing import Optional

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
import weave
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertPreTrainedModel,
    PreTrainedTokenizerFast,
)

from medrag_multi_modal.retrieval.common import (
    SimilarityMetric,
    argsort_scores,
    save_vector_index,
)
from medrag_multi_modal.utils import get_torch_backend, get_wandb_artifact


class MedCPTRetriever(weave.Model):
    """
    A class to retrieve relevant text chunks using MedCPT models.

    This class provides methods to index a dataset of text chunks and retrieve the most relevant
    chunks for a given query using MedCPT models. It uses separate models for encoding queries
    and articles, and supports both cosine similarity and Euclidean distance as similarity metrics.

    Args:
        query_encoder_model_name (str): The name of the model used for encoding queries.
        article_encoder_model_name (str): The name of the model used for encoding articles.
        chunk_size (Optional[int]): The maximum length of text chunks.
        vector_index (Optional[torch.Tensor]): The vector index of encoded text chunks.
        chunk_dataset (Optional[list[dict]]): The dataset of text chunks.
    """

    query_encoder_model_name: str
    article_encoder_model_name: str
    chunk_size: Optional[int]
    _chunk_dataset: Optional[list[dict]]
    _query_tokenizer: PreTrainedTokenizerFast
    _article_tokenizer: PreTrainedTokenizerFast
    _query_encoder_model: BertPreTrainedModel
    _article_encoder_model: BertPreTrainedModel
    _vector_index: Optional[torch.Tensor]

    def __init__(
        self,
        query_encoder_model_name: str,
        article_encoder_model_name: str,
        chunk_size: Optional[int] = None,
        vector_index: Optional[torch.Tensor] = None,
        chunk_dataset: Optional[list[dict]] = None,
    ):
        super().__init__(
            query_encoder_model_name=query_encoder_model_name,
            article_encoder_model_name=article_encoder_model_name,
            chunk_size=chunk_size,
        )
        self._query_tokenizer = AutoTokenizer.from_pretrained(
            self.query_encoder_model_name
        )
        self._article_tokenizer = AutoTokenizer.from_pretrained(
            self.article_encoder_model_name
        )
        self._query_encoder_model = AutoModel.from_pretrained(
            self.query_encoder_model_name
        )
        self._article_encoder_model = AutoModel.from_pretrained(
            self.article_encoder_model_name
        )
        self._chunk_dataset = chunk_dataset
        self._vector_index = vector_index

    def index(self, chunk_dataset_name: str, index_name: Optional[str] = None):
        """
        Indexes a dataset of text chunks and optionally saves the vector index.

        This method retrieves a dataset of text chunks from a Weave reference, encodes the text
        chunks using the article encoder model, and stores the resulting vector index. If an
        index name is provided, the vector index is saved to a file using the `save_vector_index`
        function.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            import wandb
            from medrag_multi_modal.retrieval import MedCPTRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            wandb.init(project="medrag-multi-modal", entity="ml-colabs", job_type="medcpt-index")
            retriever = MedCPTRetriever(
                query_encoder_model_name="ncbi/MedCPT-Query-Encoder",
                article_encoder_model_name="ncbi/MedCPT-Article-Encoder",
            )
            retriever.index(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_name="grays-anatomy-medcpt",
            )
            ```

        Args:
            chunk_dataset_name (str): The name of the dataset containing text chunks to be indexed.
            index_name (Optional[str]): The name to use when saving the vector index. If not provided,
                the vector index is not saved.

        """
        self._chunk_dataset = weave.ref(chunk_dataset_name).get().rows
        corpus = [row["text"] for row in self._chunk_dataset]
        with torch.no_grad():
            encoded = self._article_tokenizer(
                corpus,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.chunk_size,
            )
            vector_index = (
                self._article_encoder_model(**encoded)
                .last_hidden_state[:, 0, :]
                .contiguous()
            )
            self._vector_index = vector_index
            if index_name:
                save_vector_index(
                    self._vector_index,
                    "medcpt-index",
                    index_name,
                    {
                        "query_encoder_model_name": self.query_encoder_model_name,
                        "article_encoder_model_name": self.article_encoder_model_name,
                        "chunk_size": self.chunk_size,
                    },
                )

    @classmethod
    def from_wandb_artifact(cls, chunk_dataset_name: str, index_artifact_address: str):
        """
        Initializes an instance of the class from a Weave artifact.

        This method retrieves a precomputed vector index and its associated metadata from a Weave artifact
        stored in Weights & Biases (wandb). It then loads the vector index into memory and initializes an
        instance of the class with the retrieved model names, vector index, and chunk dataset.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            import wandb
            from medrag_multi_modal.retrieval import MedCPTRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = MedCPTRetriever.from_wandb_artifact(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-medcpt:v0",
            )
            ```

        Args:
            chunk_dataset_name (str): The name of the dataset containing text chunks to be indexed.
            index_artifact_address (str): The address of the Weave artifact containing the precomputed vector index.

        Returns:
            An instance of the class initialized with the retrieved model name, vector index, and chunk dataset.
        """
        artifact_dir, metadata = get_wandb_artifact(
            index_artifact_address, "medcpt-index", get_metadata=True
        )
        with safetensors.torch.safe_open(
            os.path.join(artifact_dir, "vector_index.safetensors"), framework="pt"
        ) as f:
            vector_index = f.get_tensor("vector_index")
        device = torch.device(get_torch_backend())
        vector_index = vector_index.to(device)
        chunk_dataset = [dict(row) for row in weave.ref(chunk_dataset_name).get().rows]
        return cls(
            query_encoder_model_name=metadata["query_encoder_model_name"],
            article_encoder_model_name=metadata["article_encoder_model_name"],
            chunk_size=metadata["chunk_size"],
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

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 2.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring. Defaults to cosine similarity.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        query = [query]
        device = torch.device(get_torch_backend())
        with torch.no_grad():
            encoded = self._query_tokenizer(
                query,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            query_embedding = self._query_encoder_model(**encoded).last_hidden_state[
                :, 0, :
            ]
            query_embedding = query_embedding.to(device)
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
        Predicts the most relevant chunks for a given query.

        This function uses the `retrieve` method to find the top-k relevant chunks
        from the dataset based on the input query. It allows specifying the number
        of top relevant chunks to retrieve and the similarity metric to use for scoring.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            import wandb
            from medrag_multi_modal.retrieval import MedCPTRetriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = MedCPTRetriever.from_wandb_artifact(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-medcpt:v0",
            )
            retriever.predict(query="What are Ribosomes?")
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 2.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring. Defaults to cosine similarity.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        return self.retrieve(query, top_k, metric)
