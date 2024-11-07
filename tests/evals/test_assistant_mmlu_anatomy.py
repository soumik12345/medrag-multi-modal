import asyncio

import weave

from medrag_multi_modal.assistant import LLMClient, MedQAAssistant
from medrag_multi_modal.metrics import MMLUOptionAccuracy
from medrag_multi_modal.retrieval.text_retrieval import (
    BM25sRetriever,
    ContrieverRetriever,
    MedCPTRetriever,
    NVEmbed2Retriever,
)


def test_mmlu_correctness_anatomy_bm25s():
    weave.init("ml-colabs/medrag-multi-modal")
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
    dataset = weave.ref("mmlu-anatomy-test:v2").get()
    evaluation = weave.Evaluation(dataset=dataset, scorers=[MMLUOptionAccuracy()])
    summary = asyncio.run(evaluation.evaluate(medqa_assistant))
    assert (
        summary["MMLUOptionAccuracy"]["correct"]["true_count"]
        > summary["MMLUOptionAccuracy"]["correct"]["false_count"]
    )


def test_mmlu_correctness_anatomy_contriever():
    weave.init("ml-colabs/medrag-multi-modal")
    retriever = ContrieverRetriever().from_index(
        index_repo_id="ashwiniai/medrag-text-corpus-chunks-contriever"
    )
    llm_client = LLMClient(model_name="gemini-1.5-flash")
    medqa_assistant = MedQAAssistant(
        llm_client=llm_client,
        retriever=retriever,
        top_k_chunks_for_query=5,
        top_k_chunks_for_options=3,
    )
    dataset = weave.ref("mmlu-anatomy-test:v2").get()
    evaluation = weave.Evaluation(dataset=dataset, scorers=[MMLUOptionAccuracy()])
    summary = asyncio.run(evaluation.evaluate(medqa_assistant))
    assert (
        summary["MMLUOptionAccuracy"]["correct"]["true_count"]
        > summary["MMLUOptionAccuracy"]["correct"]["false_count"]
    )


def test_mmlu_correctness_anatomy_medcpt():
    weave.init("ml-colabs/medrag-multi-modal")
    retriever = MedCPTRetriever().from_index(
        index_repo_id="ashwiniai/medrag-text-corpus-chunks-medcpt"
    )
    llm_client = LLMClient(model_name="gemini-1.5-flash")
    medqa_assistant = MedQAAssistant(
        llm_client=llm_client,
        retriever=retriever,
        top_k_chunks_for_query=5,
        top_k_chunks_for_options=3,
    )
    dataset = weave.ref("mmlu-anatomy-test:v2").get()
    evaluation = weave.Evaluation(dataset=dataset, scorers=[MMLUOptionAccuracy()])
    summary = asyncio.run(evaluation.evaluate(medqa_assistant))
    assert (
        summary["MMLUOptionAccuracy"]["correct"]["true_count"]
        > summary["MMLUOptionAccuracy"]["correct"]["false_count"]
    )


def test_mmlu_correctness_anatomy_nvembed2():
    weave.init("ml-colabs/medrag-multi-modal")
    retriever = NVEmbed2Retriever().from_index(
        index_repo_id="ashwiniai/medrag-text-corpus-chunks-nvembed2"
    )
    llm_client = LLMClient(model_name="gemini-1.5-flash")
    medqa_assistant = MedQAAssistant(
        llm_client=llm_client,
        retriever=retriever,
        top_k_chunks_for_query=5,
        top_k_chunks_for_options=3,
    )
    dataset = weave.ref("mmlu-anatomy-test:v2").get()
    evaluation = weave.Evaluation(dataset=dataset, scorers=[MMLUOptionAccuracy()])
    summary = asyncio.run(evaluation.evaluate(medqa_assistant))
    assert (
        summary["MMLUOptionAccuracy"]["correct"]["true_count"]
        > summary["MMLUOptionAccuracy"]["correct"]["false_count"]
    )
