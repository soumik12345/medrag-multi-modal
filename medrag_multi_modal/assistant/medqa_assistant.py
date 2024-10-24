from typing import Optional

import weave

from ..retrieval import SimilarityMetric
from .figure_annotation import FigureAnnotatorFromPageImage
from .llm_client import LLMClient


class MedQAAssistant(weave.Model):
    """Cuming"""

    llm_client: LLMClient
    retriever: weave.Model
    figure_annotator: FigureAnnotatorFromPageImage
    top_k_chunks: int = 2
    retrieval_similarity_metric: SimilarityMetric = SimilarityMetric.COSINE

    @weave.op()
    def predict(self, query: str, image_artifact_address: Optional[str] = None) -> str:
        retrieved_chunks = self.retriever.predict(
            query, top_k=self.top_k_chunks, metric=self.retrieval_similarity_metric
        )

        retrieved_chunk_texts = []
        page_indices = set()
        for chunk in retrieved_chunks:
            retrieved_chunk_texts.append(chunk["text"])
            page_indices.add(int(chunk["page_idx"]))

        figure_descriptions = []
        if image_artifact_address is not None:
            for page_idx in page_indices:
                figure_annotations = self.figure_annotator.predict(
                    page_idx=page_idx, image_artifact_address=image_artifact_address
                )
                figure_descriptions += [
                    item["figure_description"] for item in figure_annotations[page_idx]
                ]

        system_prompt = """
        You are an expert in medical science. You are given a query and a list of chunks from a medical document.
        """
        response = self.llm_client.predict(
            system_prompt=system_prompt,
            user_prompt=[query, *retrieved_chunk_texts, *figure_descriptions],
        )
        page_numbers = ", ".join([str(int(page_idx) + 1) for page_idx in page_indices])
        response += f"\n\n**Source:** {'Pages' if len(page_numbers) > 1 else 'Page'} {page_numbers} from Gray's Anatomy"
        return response
