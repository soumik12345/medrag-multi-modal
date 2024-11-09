import streamlit as st

from medrag_multi_modal.assistant import LLMClient, MedQAAssistant
from medrag_multi_modal.retrieval.text_retrieval import (
    BM25sRetriever,
    ContrieverRetriever,
    MedCPTRetriever,
    NVEmbed2Retriever,
)

# Define constants
ALL_AVAILABLE_MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gpt-4o",
    "gpt-4o-mini",
]

# Sidebar for configuration settings
st.sidebar.title("Configuration Settings")
project_name = st.sidebar.text_input(
    label="Project Name",
    value="ml-colabs/medrag-multi-modal",
    placeholder="wandb project name",
    help="format: wandb_username/wandb_project_name",
)
chunk_dataset_id = st.sidebar.selectbox(
    label="Chunk Dataset ID",
    options=["ashwiniai/medrag-text-corpus-chunks"],
)
llm_model = st.sidebar.selectbox(
    label="LLM Model",
    options=ALL_AVAILABLE_MODELS,
)
top_k_chunks_for_query = st.sidebar.slider(
    label="Top K Chunks for Query",
    min_value=1,
    max_value=20,
    value=5,
)
top_k_chunks_for_options = st.sidebar.slider(
    label="Top K Chunks for Options",
    min_value=1,
    max_value=20,
    value=3,
)
rely_only_on_context = st.sidebar.checkbox(
    label="Rely Only on Context",
    value=False,
)
retriever_type = st.sidebar.selectbox(
    label="Retriever Type",
    options=[
        "",
        "BM25S",
        "Contriever",
        "MedCPT",
        "NV-Embed-v2",
    ],
)

if retriever_type != "":

    llm_model = LLMClient(model_name=llm_model)

    retriever = None

    if retriever_type == "BM25S":
        retriever = BM25sRetriever.from_index(
            index_repo_id="ashwiniai/medrag-text-corpus-chunks-bm25s"
        )
    elif retriever_type == "Contriever":
        retriever = ContrieverRetriever.from_index(
            index_repo_id="ashwiniai/medrag-text-corpus-chunks-contriever",
            chunk_dataset=chunk_dataset_id,
        )
    elif retriever_type == "MedCPT":
        retriever = MedCPTRetriever.from_index(
            index_repo_id="ashwiniai/medrag-text-corpus-chunks-medcpt",
            chunk_dataset=chunk_dataset_id,
        )
    elif retriever_type == "NV-Embed-v2":
        retriever = NVEmbed2Retriever.from_index(
            index_repo_id="ashwiniai/medrag-text-corpus-chunks-nv-embed-2",
            chunk_dataset=chunk_dataset_id,
        )

    medqa_assistant = MedQAAssistant(
        llm_client=llm_model,
        retriever=retriever,
        top_k_chunks_for_query=top_k_chunks_for_query,
        top_k_chunks_for_options=top_k_chunks_for_options,
    )

    with st.chat_message("assistant"):
        st.markdown(
            """
Hi! I am Medrag, your medical assistant. You can ask me any questions about the medical and the life sciences.
I am currently a work-in-progress, so please bear with my stupidity and overall lack of knowledge.

**Note:** that I am not a medical professional, so please do not rely on my answers for medical decisions.
Please consult a medical professional for any medical advice.

In order to learn more about how I am being developed, please visit [soumik12345/medrag-multi-modal](https://github.com/soumik12345/medrag-multi-modal).
            """,
            unsafe_allow_html=True,
        )
    query = st.chat_input("Enter your question here")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        response = medqa_assistant.predict(query=query)
        with st.chat_message("assistant"):
            st.markdown(response.response)
