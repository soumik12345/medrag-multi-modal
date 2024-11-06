import pytest

from medrag_multi_modal.assistant import LLMClient, MedQAAssistant
from medrag_multi_modal.retrieval.text_retrieval import BM25sRetriever


@pytest.mark.retry(max_attempts=5)
def test_medqa_assistant():
    retriever = BM25sRetriever().from_index(
        index_repo_id="ashwiniai/medrag-text-corpus-chunks-bm25s"
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
        rely_only_on_context=True,
    )
    assert response.response.answer in options
    assert response.response.answer.lower() == options[2].lower()
