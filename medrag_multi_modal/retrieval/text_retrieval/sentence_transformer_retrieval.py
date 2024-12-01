import json
import os
import shutil
from abc import abstractmethod
from typing import Optional, Union

import huggingface_hub
import safetensors
import streamlit as st
import torch
import torch.nn.functional as F
import weave
from datasets import Dataset, load_dataset
from rich.progress import track
from sentence_transformers import SentenceTransformer

from medrag_multi_modal.retrieval.common import SimilarityMetric, argsort_scores
from medrag_multi_modal.utils import (
    fetch_from_huggingface,
    get_torch_backend,
    save_to_huggingface,
)


class SentenceTransformerRetriever(weave.Model):
    """
    The `SentenceTransformerRetriever` class leverages a [SentenceTransformer](https://sbert.net/)
    model to encode text chunks into vector representations and performs similarity-based retrieval.
    It supports indexing a dataset of text chunks, saving the vector index, and retrieving the most
    relevant chunks for a given query.

    Args:
        model_name (str): The name of the pre-trained model to use for encoding.
        vector_index (Optional[torch.Tensor]): The tensor containing the vector representations of
            the indexed chunks.
        chunk_dataset (Optional[list[dict]]): The dataset of text chunks to be indexed.
    """

    model_name: str
    _chunk_dataset: Optional[list[dict]]
    _model: SentenceTransformer
    _vector_index: Optional[torch.Tensor]

    def __init__(
        self,
        model_name: str = "nvidia/NV-Embed-v2",
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

    def add_end_of_sequence_tokens(self, input_examples):
        input_examples = [
            input_example + self._model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def index(
        self,
        chunk_dataset: Union[str, Dataset],
        index_repo_id: Optional[str] = None,
        cleanup: bool = True,
        batch_size: int = 8,
        streamlit_mode: bool = False,
    ):
        """
        Indexes a dataset of text chunks and optionally saves the vector index to a Huggingface repository.

        This method retrieves a dataset of text chunks from a specified source, encodes the
        text chunks into vector representations using the NV-Embed-v2 model, and stores the
        resulting vector index. If an index repository ID is provided, the vector index is saved to
        a file in the safetensors format within the specified Huggingface repository.

        !!! example "Example Usage"
            ```python
            from medrag_multi_modal.retrieval.text_retrieval import SentenceTransformerRetriever

            retriever = SentenceTransformerRetriever()
            retriever.index(
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-nv-embed-2",
            )
            ```

        ??? note "Optional Speedup using Flash Attention"
            If you have a GPU with Flash Attention support, you can enable it for NV-Embed-v2 by simply
            installing the `flash-attn` package.

            ```bash
            uv pip install flash-attn --no-build-isolation
            ```

        Args:
            chunk_dataset (str): The Huggingface dataset containing the text chunks to be indexed. Either a
                dataset repository name or a dataset object can be provided.
            index_repo_id (Optional[str]): The Huggingface repository of the index artifact to be saved.
            cleanup (bool, optional): Whether to delete the local index directory after saving the vector index.
            batch_size (int, optional): The batch size to use for encoding the corpus.
            streamlit_mode (bool): Whether or not this function is being called inside a streamlit app or not.
        """
        self._chunk_dataset = (
            load_dataset(chunk_dataset, split="chunks")
            if isinstance(chunk_dataset, str)
            else chunk_dataset
        )
        corpus = [row["text"] for row in self._chunk_dataset]
        vector_indices = []
        streamlit_progressbar = (
            st.progress(
                0,
                text="Indexing batches",
            )
            if streamlit_mode and batch_size > 1
            else None
        )
        batch_idx = 1

        for idx in track(
            range(0, len(corpus), batch_size),
            description="Encoding corpus using NV-Embed-v2",
        ):
            batch = corpus[idx : idx + batch_size]
            batch_embeddings = self._model.encode(
                self.add_end_of_sequence_tokens(batch),
                batch_size=len(batch),
                normalize_embeddings=True,
            )
            vector_indices.append(torch.tensor(batch_embeddings))
            if streamlit_progressbar:
                progress_percentage = min(
                    100, max(0, int(((idx + batch_size) / len(corpus)) * 100))
                )
                total = (len(corpus) // batch_size) + 1
                streamlit_progressbar.progress(
                    progress_percentage,
                    text=f"Indexing batch ({batch_idx}/{total})",
                )
                batch_idx += 1

        self._vector_index = torch.cat(vector_indices, dim=0)
        with torch.no_grad():
            if index_repo_id:
                index_save_dir = os.path.join(
                    ".huggingface", index_repo_id.split("/")[-1]
                )
                os.makedirs(index_save_dir, exist_ok=True)
                safetensors.torch.save_file(
                    {"vector_index": self._vector_index.cpu()},
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
    def from_index(
        cls,
        chunk_dataset: Union[str, Dataset],
        index_repo_id: str,
        chunk_dataset_split: Optional[str] = None,
    ):
        """
        Creates an instance of the class from a Huggingface repository.

        This method retrieves a vector index and metadata from a Huggingface repository. It also retrieves a dataset of text chunks from a Huggingface dataset repository. The vector index is loaded from a safetensors file and moved to the appropriate device (CPU or GPU). The text chunks are converted into a list of dictionaries. The method then returns an instance of the class initialized with the retrieved model name, vector index, and chunk dataset.
        Weights & Biases (wandb). It also retrieves a dataset of text chunks from a Weave
        reference. The vector index is loaded from a safetensors file and moved to the
        appropriate device (CPU or GPU). The text chunks are converted into a list of
        dictionaries. The method then returns an instance of the class initialized with
        the retrieved model name, vector index, and chunk dataset.

        !!! example "Example Usage"
            ```python
            import weave
            from medrag_multi_modal.retrieval.text_retrieval import SentenceTransformerRetriever

            retriever = SentenceTransformerRetriever.from_index(
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-nv-embed-2",
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
            )
            ```

        ??? note "Optional Speedup using Flash Attention"
            If you have a GPU with Flash Attention support, you can enable it for NV-Embed-v2 by simply
            installing the `flash-attn` package.

            ```bash
            uv pip install flash-attn --no-build-isolation
            ```

        Args:
            chunk_dataset (str): The Huggingface dataset containing the text chunks to be indexed. Either a
                dataset repository name or a dataset object can be provided.
            index_repo_id (str): The Huggingface repository of the index artifact to be saved.
            chunk_dataset_split (Optional[str]): The split of the dataset to be indexed.

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
            load_dataset(chunk_dataset, split=chunk_dataset_split)
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
            from medrag_multi_modal.retrieval.text_retrieval import SentenceTransformerRetriever

            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = SentenceTransformerRetriever.from_index(
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-nv-embed-2",
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
            )
            retriever.retrieve(query="What is ribosome?")
            ```

        ??? note "Optional Speedup using Flash Attention"
            If you have a GPU with Flash Attention support, you can enable it for NV-Embed-v2 by simply
            installing the `flash-attn` package.

            ```bash
            uv pip install flash-attn --no-build-isolation
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
                self.add_end_of_sequence_tokens(query), normalize_embeddings=True
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
                    **self._chunk_dataset[score["original_index"]],
                    **{"score": score["item"]},
                }
            )
        return retrieved_chunks

    @abstractmethod
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
            from medrag_multi_modal.retrieval.text_retrieval import SentenceTransformerRetriever

            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = SentenceTransformerRetriever.from_index(
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-nv-embed-2",
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
            )
            retriever.predict(query="What is ribosome?")
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        return self.retrieve(query, top_k, metric)


class NVEmbed2Retriever(SentenceTransformerRetriever):
    """
    `NVEmbed2Retriever` is a class for retrieving relevant text chunks from a dataset using the
    [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) model.

    This class leverages the SentenceTransformer model to encode text chunks into vector representations and
    performs similarity-based retrieval. It supports indexing a dataset of text chunks, saving the vector index,
    and retrieving the most relevant chunks for a given query.

    Args:
        vector_index (Optional[torch.Tensor]): The tensor containing the vector representations of the indexed chunks.
        chunk_dataset (Optional[list[dict]]): The dataset of text chunks to be indexed.
    """

    def __init__(
        self,
        vector_index: Optional[torch.Tensor] = None,
        chunk_dataset: Optional[list[dict]] = None,
    ):
        super().__init__(
            self,
            model_name="nvidia/NV-Embed-v2",
            vector_index=vector_index,
            chunk_dataset=chunk_dataset,
        )

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
            from medrag_multi_modal.retrieval.text_retrieval import SentenceTransformerRetriever

            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = NVEmbed2Retriever.from_index(
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-nv-embed-2",
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
            )
            retriever.predict(query="What is ribosome?")
            ```

        ??? note "Optional Speedup using Flash Attention"
            If you have a GPU with Flash Attention support, you can enable it for NV-Embed-v2 by simply
            installing the `flash-attn` package.

            ```bash
            uv pip install flash-attn --no-build-isolation
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        query = [
            f"""Instruct: Given a question, retrieve passages that answer the question
Query: {query}"""
        ]
        return self.retrieve(query, top_k, metric)
