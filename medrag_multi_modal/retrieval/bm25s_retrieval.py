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

    def index(self, corpus_dataset_name: str, index_name: Optional[str] = None):
        corpus_dataset = weave.ref(corpus_dataset_name).get().rows
        corpus = [row["text"] for row in corpus_dataset]
        corpus_tokens = bm25s.tokenize(
            corpus,
            stopwords=LANGUAGE_DICT[self.language],
            stemmer=Stemmer(self.language) if self.use_stemmer else None,
        )
        self._retriever.index(corpus_tokens)
        if index_name:
            self._retriever.save(
                index_name, corpus=[dict(row) for row in corpus_dataset]
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
        query_tokens = bm25s.tokenize(
            query,
            stopwords=LANGUAGE_DICT[self.language],
            stemmer=Stemmer(self.language) if self.use_stemmer else None,
        )
        results, scores = self._retriever.retrieve(query_tokens, k=top_k)
        return {
            "results": results,
            "scores": scores,
        }
