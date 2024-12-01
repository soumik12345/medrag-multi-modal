import json
import os
import shutil
from typing import Optional, Union

import huggingface_hub
import safetensors
import safetensors.torch
import streamlit as st
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

from medrag_multi_modal.retrieval.common import SimilarityMetric, argsort_scores
from medrag_multi_modal.utils import (
    fetch_from_huggingface,
    get_torch_backend,
    save_to_huggingface,
)


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
        query_encoder_model_name: str = "ncbi/MedCPT-Query-Encoder",
        article_encoder_model_name: str = "ncbi/MedCPT-Article-Encoder",
        chunk_size: Optional[int] = 512,
        vector_index: Optional[torch.Tensor] = None,
        chunk_dataset: Optional[list[dict]] = None,
    ):
        super().__init__(
            query_encoder_model_name=query_encoder_model_name,
            article_encoder_model_name=article_encoder_model_name,
            chunk_size=chunk_size,
        )
        self._query_tokenizer = AutoTokenizer.from_pretrained(
            self.query_encoder_model_name, max_length=self.chunk_size
        )
        self._article_tokenizer = AutoTokenizer.from_pretrained(
            self.article_encoder_model_name, max_length=self.chunk_size
        )
        self._query_encoder_model = AutoModel.from_pretrained(
            self.query_encoder_model_name
        ).to(get_torch_backend())
        self._article_encoder_model = AutoModel.from_pretrained(
            self.article_encoder_model_name
        ).to(get_torch_backend())
        self._chunk_dataset = chunk_dataset
        self._vector_index = vector_index

    def index(
        self,
        chunk_dataset: Union[str, Dataset],
        chunk_dataset_split: str,
        index_repo_id: Optional[str] = None,
        cleanup: bool = True,
        batch_size: int = 32,
        streamlit_mode: bool = False,
    ):
        """
        Indexes a dataset of text chunks using the MedCPT model and optionally saves the vector index.

        This method retrieves a dataset of text chunks from a specified source, encodes the text
        chunks into vector representations using the article encoder model, and stores the
        resulting vector index. If an `index_repo_id` is provided, the vector index is saved
        to disk in the safetensors format and optionally logged as a Huggingface artifact.

        !!! example "Example Usage"
            ```python
            import weave
            from dotenv import load_dotenv

            from medrag_multi_modal.retrieval.text_retrieval import MedCPTRetriever

            load_dotenv()
            retriever = MedCPTRetriever()
            retriever.index(
                chunk_dataset="geekyrakshit/grays-anatomy-chunks-test",
                index_repo_id="geekyrakshit/grays-anatomy-index-medcpt",
            )
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
            load_dataset(chunk_dataset, split=chunk_dataset_split)
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
        with torch.no_grad():
            for idx in track(
                range(0, len(corpus), batch_size),
                description="Encoding corpus using MedCPT",
            ):
                batch = corpus[idx : idx + batch_size]
                encoded = self._article_tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.chunk_size,
                ).to(get_torch_backend())
                batch_vectors = (
                    self._article_encoder_model(**encoded)
                    .last_hidden_state[:, 0, :]
                    .contiguous()
                )
                vector_indices.append(batch_vectors)
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

            vector_index = torch.cat(vector_indices, dim=0)
            self._vector_index = vector_index
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
                        {
                            "query_encoder_model_name": self.query_encoder_model_name,
                            "article_encoder_model_name": self.article_encoder_model_name,
                            "chunk_size": self.chunk_size,
                        },
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

        This method retrieves a vector index and metadata from a Huggingface repository.
        It also retrieves a dataset of text chunks from the specified source. The vector
        index is loaded from a safetensors file and moved to the appropriate device (CPU or GPU).
        The method then returns an instance of the class initialized with the retrieved
        model names, vector index, and chunk dataset.

        !!! example "Example Usage"
            ```python
            from medrag_multi_modal.retrieval.text_retrieval import MedCPTRetriever

            retriever = MedCPTRetriever.from_index(
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-medcpt",
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
            )
            ```

        Args:
            chunk_dataset (str): The Huggingface dataset containing the text chunks to be indexed. Either a
                dataset repository name or a dataset object can be provided.
            index_repo_id (str): The Huggingface repository of the index artifact to be saved.
            chunk_dataset_split (Optional[str]): The split of the dataset to be indexed.

        Returns:
            An instance of the class initialized with the retrieved model name, vector index, and chunk dataset.
        """
        index_dir = fetch_from_huggingface(index_repo_id, ".huggingface")
        with safetensors.torch.safe_open(
            os.path.join(index_dir, "vector_index.safetensors"), framework="pt"
        ) as f:
            vector_index = f.get_tensor("vector_index")
        device = torch.device(get_torch_backend())
        vector_index = vector_index.to(device)
        with open(os.path.join(index_dir, "config.json"), "r") as config_file:
            metadata = json.load(config_file)
        chunk_dataset = (
            load_dataset(chunk_dataset, split=chunk_dataset_split)
            if isinstance(chunk_dataset, str)
            else chunk_dataset
        )
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

        !!! example "Example Usage"
            ```python
            import weave
            from medrag_multi_modal.retrieval.text_retrieval import MedCPTRetriever

            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = MedCPTRetriever.from_index(
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-medcpt",
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
            )
            retriever.retrieve(query="What is ribosome?")
            ```

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
            ).to(device)
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
            from medrag_multi_modal.retrieval.text_retrieval import MedCPTRetriever

            weave.init(project_name="ml-colabs/medrag-multi-modal")
            retriever = MedCPTRetriever.from_index(
                index_repo_id="ashwiniai/medrag-text-corpus-chunks-medcpt",
                chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
            )
            retriever.predict(query="What is ribosome?")
            ```

        Args:
            query (str): The input query string to search for relevant chunks.
            top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 2.
            metric (SimilarityMetric, optional): The similarity metric to use for scoring. Defaults to cosine similarity.

        Returns:
            list: A list of dictionaries, each containing a retrieved chunk and its relevance score.
        """
        return self.retrieve(query, top_k, metric)
