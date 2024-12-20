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


def test_mmlu_correctness_anatomy_bm25s(model_name: str):
    weave.init("ml-colabs/medrag-multi-modal")
    retriever = BM25sRetriever().from_index(
        index_repo_id="ashwiniai/medrag-text-corpus-chunks-bm25s"
    )
    llm_client = LLMClient(model_name=model_name)
    medqa_assistant = MedQAAssistant(
        llm_client=llm_client,
        retriever=retriever,
        top_k_chunks_for_query=5,
        top_k_chunks_for_options=3,
    )
    dataset = weave.ref("mmlu-anatomy-test:v2").get()
    with weave.attributes(
        {"retriever": retriever.__class__.__name__, "llm": llm_client.model_name}
    ):
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=[MMLUOptionAccuracy()],
            name="MMLU-Anatomy-BM25s",
        )
        summary = asyncio.run(
            evaluation.evaluate(
                medqa_assistant,
                __weave={"display_name": evaluation.name + ":" + llm_client.model_name},
            )
        )
    assert (
        summary["MMLUOptionAccuracy"]["correct"]["true_count"]
        > summary["MMLUOptionAccuracy"]["correct"]["false_count"]
    )


def test_mmlu_correctness_anatomy_contriever(model_name: str):
    weave.init("ml-colabs/medrag-multi-modal")
    retriever = ContrieverRetriever().from_index(
        index_repo_id="ashwiniai/medrag-text-corpus-chunks-contriever",
        chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
    )
    llm_client = LLMClient(model_name=model_name)
    medqa_assistant = MedQAAssistant(
        llm_client=llm_client,
        retriever=retriever,
        top_k_chunks_for_query=5,
        top_k_chunks_for_options=3,
    )
    dataset = weave.ref("mmlu-anatomy-test:v2").get()
    with weave.attributes(
        {"retriever": retriever.__class__.__name__, "llm": llm_client.model_name}
    ):
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=[MMLUOptionAccuracy()],
            name="MMLU-Anatomy-Contriever",
        )
        summary = asyncio.run(
            evaluation.evaluate(
                medqa_assistant,
                __weave={"display_name": evaluation.name + ":" + llm_client.model_name},
            )
        )
    assert (
        summary["MMLUOptionAccuracy"]["correct"]["true_count"]
        > summary["MMLUOptionAccuracy"]["correct"]["false_count"]
    )


def test_mmlu_correctness_anatomy_medcpt(model_name: str):
    weave.init("ml-colabs/medrag-multi-modal")
    retriever = MedCPTRetriever().from_index(
        index_repo_id="ashwiniai/medrag-text-corpus-chunks-medcpt",
        chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
    )
    llm_client = LLMClient(model_name=model_name)
    medqa_assistant = MedQAAssistant(
        llm_client=llm_client,
        retriever=retriever,
        top_k_chunks_for_query=5,
        top_k_chunks_for_options=3,
    )
    dataset = weave.ref("mmlu-anatomy-test:v2").get()
    with weave.attributes(
        {"retriever": retriever.__class__.__name__, "llm": llm_client.model_name}
    ):
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=[MMLUOptionAccuracy()],
            name="MMLU-Anatomy-MedCPT",
        )
        summary = asyncio.run(
            evaluation.evaluate(
                medqa_assistant,
                __weave={"display_name": evaluation.name + ":" + llm_client.model_name},
            )
        )
    assert (
        summary["MMLUOptionAccuracy"]["correct"]["true_count"]
        > summary["MMLUOptionAccuracy"]["correct"]["false_count"]
    )


def test_mmlu_correctness_anatomy_nvembed2(model_name: str):
    weave.init("ml-colabs/medrag-multi-modal")
    retriever = NVEmbed2Retriever().from_index(
        index_repo_id="ashwiniai/medrag-text-corpus-chunks-nv-embed-2",
        chunk_dataset="ashwiniai/medrag-text-corpus-chunks",
    )
    llm_client = LLMClient(model_name=model_name)
    medqa_assistant = MedQAAssistant(
        llm_client=llm_client,
        retriever=retriever,
        top_k_chunks_for_query=5,
        top_k_chunks_for_options=3,
    )
    dataset = weave.ref("mmlu-anatomy-test:v2").get()
    with weave.attributes(
        {"retriever": retriever.__class__.__name__, "llm": llm_client.model_name}
    ):
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=[MMLUOptionAccuracy()],
            name="MMLU-Anatomy-NVEmbed2",
        )
        summary = asyncio.run(
            evaluation.evaluate(
                medqa_assistant,
                __weave={"display_name": evaluation.name + ":" + llm_client.model_name},
            )
        )
    assert (
        summary["MMLUOptionAccuracy"]["correct"]["true_count"]
        > summary["MMLUOptionAccuracy"]["correct"]["false_count"]
    )
