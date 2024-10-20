from enum import Enum

import wandb


class SimilarityMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_wandb_artifact(artifact_address: str, artifact_type: str):
    if wandb.run:
        artifact = wandb.run.use_artifact(artifact_address, type=artifact_type)
        artifact_dir = artifact.download()
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact_address)
        artifact_dir = artifact.download()
    metadata = artifact.metadata
    return artifact_dir, metadata


def argsort_scores(scores: list[float], descending: bool = False):
    return [
        {"item": item, "original_index": idx}
        for idx, item in sorted(
            list(enumerate(scores)), key=lambda x: x[1], reverse=descending
        )
    ]
