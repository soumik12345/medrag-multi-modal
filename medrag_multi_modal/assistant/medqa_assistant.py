import weave

from medrag_multi_modal.assistant.figure_annotation import FigureAnnotatorFromPageImage
from medrag_multi_modal.assistant.llm_client import LLMClient
from medrag_multi_modal.retrieval.common import SimilarityMetric


class MedQAAssistant(weave.Model):
    """
    `MedQAAssistant` is a class designed to assist with medical queries by leveraging a
    language model client, a retriever model, and a figure annotator.

    !!! example "Usage Example"
        ```python
        import weave
        from dotenv import load_dotenv

        from medrag_multi_modal.assistant import (
            FigureAnnotatorFromPageImage,
            LLMClient,
            MedQAAssistant,
        )
        from medrag_multi_modal.retrieval import MedCPTRetriever

        load_dotenv()
        weave.init(project_name="ml-colabs/medrag-multi-modal")

        llm_client = LLMClient(model_name="gemini-1.5-flash")

        retriever=MedCPTRetriever.from_wandb_artifact(
            chunk_dataset_name="grays-anatomy-chunks:v0",
            index_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-medcpt:v0",
        )

        figure_annotator=FigureAnnotatorFromPageImage(
            figure_extraction_llm_client=LLMClient(model_name="pixtral-12b-2409"),
            structured_output_llm_client=LLMClient(model_name="gpt-4o"),
            image_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-images-marker:v6",
        )
        medqa_assistant = MedQAAssistant(
            llm_client=llm_client, retriever=retriever, figure_annotator=figure_annotator
        )
        medqa_assistant.predict(query="What is ribosome?")
        ```

    Args:
        llm_client (LLMClient): The language model client used to generate responses.
        retriever (weave.Model): The model used to retrieve relevant chunks of text from a medical document.
        figure_annotator (FigureAnnotatorFromPageImage): The annotator used to extract figure descriptions from pages.
        top_k_chunks (int): The number of top chunks to retrieve based on similarity metric.
        retrieval_similarity_metric (SimilarityMetric): The metric used to measure similarity for retrieval.
    """

    llm_client: LLMClient
    retriever: weave.Model
    figure_annotator: FigureAnnotatorFromPageImage
    top_k_chunks: int = 2
    retrieval_similarity_metric: SimilarityMetric = SimilarityMetric.COSINE

    @weave.op()
    def predict(self, query: str) -> str:
        """
        Generates a response to a medical query by retrieving relevant text chunks and figure descriptions
        from a medical document and using a language model to generate the final response.

        This function performs the following steps:
        1. Retrieves relevant text chunks from the medical document based on the query using the retriever model.
        2. Extracts the text and page indices from the retrieved chunks.
        3. Retrieves figure descriptions from the pages identified in the previous step using the figure annotator.
        4. Constructs a system prompt and user prompt combining the query, retrieved text chunks, and figure descriptions.
        5. Uses the language model client to generate a response based on the constructed prompts.
        6. Appends the source information (page numbers) to the generated response.

        Args:
            query (str): The medical query to be answered.

        Returns:
            str: The generated response to the query, including source information.
        """
        retrieved_chunks = self.retriever.predict(
            query, top_k=self.top_k_chunks, metric=self.retrieval_similarity_metric
        )

        retrieved_chunk_texts = []
        page_indices = set()
        for chunk in retrieved_chunks:
            retrieved_chunk_texts.append(chunk["text"])
            page_indices.add(int(chunk["page_idx"]))

        figure_descriptions = []
        for page_idx in page_indices:
            figure_annotations = self.figure_annotator.predict(page_idx=page_idx)[
                page_idx
            ]
            figure_descriptions += [
                item["figure_description"] for item in figure_annotations
            ]

        system_prompt = """
        You are an expert in medical science. You are given a query and a list of chunks from a medical document.
        """
        response = self.llm_client.predict(
            system_prompt=system_prompt,
            user_prompt=[query, *retrieved_chunk_texts, *figure_descriptions],
        )
        page_numbers = ", ".join([str(int(page_idx) + 1) for page_idx in page_indices])
        response += f"\n\n**Source:** {'Pages' if len(page_indices) > 1 else 'Page'} {page_numbers} from Gray's Anatomy"
        return response
