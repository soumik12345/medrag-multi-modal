from enum import Enum


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
