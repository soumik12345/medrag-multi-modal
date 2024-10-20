import os
from typing import Optional

import safetensors
import torch
import torch.nn.functional as F
import weave
from sentence_transformers import SentenceTransformer

from ..utils import get_torch_backend, get_wandb_artifact
from .common import SimilarityMetric, argsort_scores, save_vector_index


class NVEmbed2Retriever(weave.Model):
    """
    `NVEmbed2Retriever` is a class for retrieving relevant text chunks from a dataset using the
    [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) model.

    This class leverages the SentenceTransformer model to encode text chunks into vector representations and
    performs similarity-based retrieval. It supports indexing a dataset of text chunks, saving the vector index,
    and retrieving the most relevant chunks for a given query.

    Args:
        model_name (str): The name of the pre-trained model to use for encoding.
        vector_index (Optional[torch.Tensor]): The tensor containing the vector representations of the indexed chunks.
        chunk_dataset (Optional[list[dict]]): The dataset of text chunks to be indexed.
    """

    model_name: str
    _chunk_dataset: Optional[list[dict]]
    _model: SentenceTransformer
    _vector_index: Optional[torch.Tensor]

    def __init__(
        self,
        model_name: str = "sentence-transformers/nvembed2-nli-v1",
        vector_index: Optional[torch.Tensor] = None,
        chunk_dataset: Optional[list[dict]] = None,
    ):
        super().__init__(model_name=model_name)
        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
            device=get_torch_backend(),
        )
        self._model.max_seq_length = 32768
        self._model.tokenizer.padding_side = "right"
        self._vector_index = vector_index
        self._chunk_dataset = chunk_dataset

    def add_eos(self, input_examples):
        input_examples = [
            input_example + self._model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def index(self, chunk_dataset_name: str, index_name: Optional[str] = None):
        """
        Indexes a dataset of text chunks and optionally saves the vector index to a file.

        This method retrieves a dataset of text chunks from a Weave reference, encodes the
        text chunks into vector representations using the NV-Embed-v2 model, and stores the
        resulting vector index. If an index name is provided, the vector index is saved to
        a file in the safetensors format. Additionally, if a Weave run is active, the vector
        index file is logged as an artifact to Weave.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            import wandb
            from medrag_multi_modal.retrieval import NVEmbed2Retriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            wandb.init(project="medrag-multi-modal", entity="ml-colabs", job_type="nvembed2-index")
            retriever = NVEmbed2Retriever(model_name="nvidia/NV-Embed-v2")
            retriever.index(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_name="grays-anatomy-nvembed2",
            )
            ```

        Args:
            chunk_dataset_name (str): The name of the Weave dataset containing the text chunks
                to be indexed.
            index_name (Optional[str]): The name of the index artifact to be saved. If provided,
                the vector index is saved to a file and logged as an artifact to Weave.
        """
        self._chunk_dataset = weave.ref(chunk_dataset_name).get().rows
        corpus = [row["text"] for row in self._chunk_dataset]
        self._vector_index = self._model.encode(
            self.add_eos(corpus), batch_size=len(corpus), normalize_embeddings=True
        )
        with torch.no_grad():
            if index_name:
                save_vector_index(
                    torch.from_numpy(self._vector_index),
                    "nvembed2-index",
                    index_name,
                    {"model_name": self.model_name},
                )

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

            import wandb
            from medrag_multi_modal.retrieval import NVEmbed2Retriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = NVEmbed2Retriever(model_name="nvidia/NV-Embed-v2")
            retriever.index(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_name="grays-anatomy-nvembed2",
            )
            retriever = NVEmbed2Retriever.from_wandb_artifact(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-nvembed2:v0",
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
            index_artifact_address, "nvembed2-index", get_metadata=True
        )
        with safetensors.torch.safe_open(
            os.path.join(artifact_dir, "vector_index.safetensors"), framework="pt"
        ) as f:
            vector_index = f.get_tensor("vector_index")
        device = torch.device(get_torch_backend())
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
        query: list[str],
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

            import wandb
            from medrag_multi_modal.retrieval import NVEmbed2Retriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = NVEmbed2Retriever(model_name="nvidia/NV-Embed-v2")
            retriever.index(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_name="grays-anatomy-nvembed2",
            )
            retriever = NVEmbed2Retriever.from_wandb_artifact(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-nvembed2:v0",
            )
            ```

        Args:
            query (list[str]): The input query strings to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        device = torch.device(get_torch_backend())
        with torch.no_grad():
            query_embedding = self._model.encode(
                self.add_eos(query), normalize_embeddings=True
            )
            query_embedding = torch.from_numpy(query_embedding).to(device)
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

    @weave.op()
    def predict(
        self,
        query: str,
        top_k: int = 2,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        """
        Predicts the top-k most relevant chunks for a given query using the specified similarity metric.

        This method formats the input query string by prepending an instruction prompt and then calls the
        `retrieve` method to get the most relevant chunks. The similarity metric can be either cosine similarity
        or Euclidean distance. The top-k chunks with the highest similarity scores are returned.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            import wandb
            from medrag_multi_modal.retrieval import NVEmbed2Retriever

            load_dotenv()
            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = NVEmbed2Retriever(model_name="nvidia/NV-Embed-v2")
            retriever.index(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_name="grays-anatomy-nvembed2",
            )
            retriever = NVEmbed2Retriever.from_wandb_artifact(
                chunk_dataset_name="grays-anatomy-chunks:v0",
                index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-nvembed2:v0",
            )
            retriever.predict(query="What are Ribosomes?")
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        query = [
            f"Instruct: Given a question, retrieve passages that answer the question\nQuery: {query}"
        ]
        return self.retrieve(query, top_k, metric)
