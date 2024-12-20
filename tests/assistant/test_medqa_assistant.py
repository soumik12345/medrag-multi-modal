import pytest

from medrag_multi_modal.assistant import LLMClient, MedQAAssistant
from medrag_multi_modal.retrieval.text_retrieval import BM25sRetriever


@pytest.mark.skip(reason="Repository not implemented")
def test_medqa_assistant():
    retriever = BM25sRetriever.from_index(
        index_repo_id="ashwiniai/anatomy-corpus-pypdf2textloader-bm25s"
    )
    llm_client = LLMClient(model_name="gemini-1.5-flash")
    medqa_assistant = MedQAAssistant(
        llm_client=llm_client,
        retriever=retriever,
        top_k_chunks_for_query=5,
        top_k_chunks_for_options=3,
    )
    options = [
        "The first pharyngeal arch",
        "The first and second pharyngeal arches",
        "The second pharyngeal arch",
        "The second and third pharyngeal arches",
    ]
    response = medqa_assistant.predict(
        query="What is the embryological origin of the hyoid bone?",
        options=options,
    )
    assert response.response.answer in options
    assert response.response.answer.lower() == options[2].lower()
