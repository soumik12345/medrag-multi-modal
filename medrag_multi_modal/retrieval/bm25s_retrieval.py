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
        self._retriever.save(index_name, corpus=[dict(row) for row in corpus_dataset])
        if index_name:
            self._retriever.save(index_name)
            if wandb.run:
                artifact = wandb.Artifact(name=index_name, type="bm25s-index")
                artifact.add_dir(index_name)
                artifact.save()
