
import weave

from ..retrieval import SimilarityMetric
from .llm_client import LLMClient


class MedQAAssistant(weave.Model):
    """Cuming"""
    llm_client: LLMClient
    retriever: weave.Model
    top_k_chunks: int = 2
    retrieval_similarity_metric: SimilarityMetric = SimilarityMetric.COSINE

    @weave.op()
    def predict(self, query: str) -> str:
        retrieved_chunks = self.retriever.predict(
            query, top_k=self.top_k_chunks, metric=self.retrieval_similarity_metric
        )
        retrieved_chunks = [chunk["text"] for chunk in retrieved_chunks]
        system_prompt = """
        You are a medical expert. You are given a query and a list of chunks from a medical document.
        """
        return self.llm_client.predict(
            system_prompt=system_prompt, user_prompt=retrieved_chunks
        )
