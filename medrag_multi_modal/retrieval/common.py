from enum import Enum

import safetensors
import safetensors.torch
import torch
import wandb


class SimilarityMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def argsort_scores(scores: list[float], descending: bool = False):
    return [
        {"item": item, "original_index": idx}
        for idx, item in sorted(
            list(enumerate(scores)), key=lambda x: x[1], reverse=descending
        )
    ]


def save_vector_index(
    vector_index: torch.Tensor,
    type: str,
    index_name: str,
    metadata: dict,
    filename: str = "vector_index.safetensors",
):
    safetensors.torch.save_file({"vector_index": vector_index.cpu()}, filename)
    if wandb.run:
        artifact = wandb.Artifact(
            name=index_name,
            type=type,
            metadata=metadata,
        )
        artifact.add_file(filename)
        artifact.save()
