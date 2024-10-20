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

import wandb

from .common import SimilarityMetric, argsort_scores, get_wandb_artifact, mean_pooling


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
        self._model = AutoModel.from_pretrained(self.model_name)
        self._vector_index = vector_index
        self._chunk_dataset = chunk_dataset

    def encode(self, corpus: list[str]) -> torch.Tensor:
        inputs = self._tokenizer(
            corpus, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self._model(**inputs)
        return mean_pooling(outputs[0], inputs["attention_mask"])

    def index(self, chunk_dataset_name: str, index_name: Optional[str] = None):
        """
        Indexes a dataset of text chunks and optionally saves the vector index to a file.

        This method retrieves a dataset of text chunks from a Weave reference, encodes the
        text chunks into vector representations using the Contriever model, and stores the
        resulting vector index. If an index name is provided, the vector index is saved to
        a file in the safetensors format. Additionally, if a Weave run is active, the vector
        index file is logged as an artifact to Weave.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            import wandb
            from medrag_multi_modal.retrieval import ContrieverRetriever, SimilarityMetric

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            wandb.init(project="medrag-multi-modal", entity="ml-colabs", job_type="contriever-index")
            retriever = ContrieverRetriever(model_name="facebook/contriever")
            retriever.index(chunk_dataset_name="grays-anatomy-chunks:v0", index_name="grays-anatomy-contriever")
            ```

        Args:
            chunk_dataset_name (str): The name of the Weave dataset containing the text chunks
                to be indexed.
            index_name (Optional[str]): The name of the index artifact to be saved. If provided,
                the vector index is saved to a file and logged as an artifact to Weave.
        """
        self._chunk_dataset = weave.ref(chunk_dataset_name).get().rows
        corpus = [row["text"] for row in self._chunk_dataset]
        with torch.no_grad():
            vector_index = self.encode(corpus)
            self._vector_index = vector_index
            if index_name:
                safetensors.torch.save_file(
                    {"vector_index": vector_index.cpu()}, "vector_index.safetensors"
                )
                if wandb.run:
                    artifact = wandb.Artifact(
                        name=index_name,
                        type="contriever-index",
                        metadata={"model_name": self.model_name},
                    )
                    artifact.add_file("vector_index.safetensors")
                    artifact.save()

    @classmethod
    def from_wandb_artifact(cls, chunk_dataset_name: str, index_artifact_address: str):
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

            from medrag_multi_modal.retrieval import ContrieverRetriever, SimilarityMetric

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = ContrieverRetriever.from_wandb_artifact(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-contriever:v1",
            )
            ```

        Args:
            chunk_dataset_name (str): The name of the Weave dataset containing the text chunks.
            index_artifact_address (str): The address of the Weave artifact containing the
                vector index.

        Returns:
            An instance of the class initialized with the retrieved model name, vector index,
            and chunk dataset.
        """
        artifact_dir, metadata = get_wandb_artifact(
            index_artifact_address, "contriever-index"
        )
        with safetensors.torch.safe_open(
            os.path.join(artifact_dir, "vector_index.safetensors"), framework="pt"
        ) as f:
            vector_index = f.get_tensor("vector_index")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vector_index = vector_index.to(device)
        chunk_dataset = [dict(row) for row in weave.ref(chunk_dataset_name).get().rows]
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

            from medrag_multi_modal.retrieval import ContrieverRetriever, SimilarityMetric

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = ContrieverRetriever.from_wandb_artifact(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-contriever:v1",
            )
            scores = retriever.retrieve(query="What are Ribosomes?", metric=SimilarityMetric.COSINE)
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 2.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        query = [query]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            query_embedding = self.encode(query).to(device)
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
                    "chunk": self._chunk_dataset[score["original_index"]],
                    "score": score["item"],
                }
            )
        return retrieved_chunks