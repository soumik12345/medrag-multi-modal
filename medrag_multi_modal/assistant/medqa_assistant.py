from typing import Optional

import weave

from medrag_multi_modal.assistant.figure_annotation import FigureAnnotatorFromPageImage
from medrag_multi_modal.assistant.llm_client import LLMClient
from medrag_multi_modal.assistant.schema import (
    MedQACitation,
    MedQAMCQResponse,
    MedQAResponse,
)
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
    def predict(
        self,
        query: str,
        options: Optional[list[str]] = None,
        rely_only_on_context: bool = True,
    ) -> str:
        """
        Generates a response to a medical query by retrieving relevant text chunks and figure descriptions
        from a medical document and using a language model to generate the final response.

        This function performs the following steps:
        1. Retrieves relevant text chunks from the medical document based on the query and any provided options
           using the retriever model.
        2. Extracts the text and page indices from the retrieved chunks.
        3. Retrieves figure descriptions from the pages identified in the previous step using the figure annotator.
        4. Constructs a system prompt and user prompt combining the query, options (if provided), retrieved text chunks,
           and figure descriptions.
        5. Uses the language model client to generate a response based on the constructed prompts, either choosing
           from provided options or generating a free-form response.
        6. Returns the generated response, which includes the answer and explanation if options were provided.

        The function can operate in two modes:
        - Multiple choice: When options are provided, it selects the best answer from the options and explains the choice
        - Free response: When no options are provided, it generates a comprehensive response based on the context

        Args:
            query (str): The medical query to be answered.
            options (Optional[list[str]]): The list of options to choose from.
            rely_only_on_context (bool): Whether to rely only on the context provided or not during response generation.

        Returns:
            str: The generated response to the query, including source information.
        """
        retrieved_chunks = self.retriever.predict(
            query, top_k=self.top_k_chunks, metric=self.retrieval_similarity_metric
        )

        options = options or []
        for option in options:
            retrieved_chunks += self.retriever.predict(
                query=option,
                top_k=self.top_k_chunks,
                metric=self.retrieval_similarity_metric,
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

        system_prompt = """You are an expert in medical science. You are given a question
and a list of excerpts from various medical documents.
        """
        query = f"""# Question
{query}
        """

        if len(options) > 0:
            system_prompt += """\nYou are also given a list of options to choose your answer from.
You are supposed to choose the best possible option based on the context provided. You should also
explain your answer to justify why you chose that option.
"""
            query += "## Options\n"
            for option in options:
                query += f"- {option}\n"
        else:
            system_prompt += "\nYou are supposed to answer the question based on the context provided."

        if rely_only_on_context:
            query += """\n\nYou are only allowed to use the context provided to answer the question.
You are not allowed to use any external knowledge to answer the question.
"""

        response = self.llm_client.predict(
            system_prompt=system_prompt,
            user_prompt=[query, *retrieved_chunk_texts, *figure_descriptions],
            schema=MedQAMCQResponse if len(options) > 0 else None,
        )

        # TODO: Add figure citations
        # TODO: Add source document name from retrieved chunks as citations
        citations = []
        for page_idx in page_indices:
            citations.append(
                MedQACitation(page_number=page_idx + 1, document_name="Gray's Anatomy")
            )

        return MedQAResponse(response=response, citations=citations)
