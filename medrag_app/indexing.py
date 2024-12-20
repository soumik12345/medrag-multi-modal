import importlib

import streamlit as st

st.session_state.start_indexing = False

st.title("💿 Indexing")

retriever_name = st.sidebar.selectbox(
    "Text Retriever",
    [
        "",
        "BM25sRetriever",
        "ContrieverRetriever",
        "MedCPTRetriever",
        "NVEmbed2Retriever",
    ],
)

if retriever_name != "":
    module_name = "medrag_multi_modal.retrieval.text_retrieval"
    retriever_class = getattr(importlib.import_module(module_name), retriever_name)
    with st.sidebar.status("Initializing retriever..."):
        retriever = retriever_class()

    if retriever_name == "MedCPTRetriever":
        chunk_size = st.sidebar.slider(
            label="Chunk Size", min_value=128, max_value=1024, value=512, step=128
        )

    chunk_dataset_repo_id = st.sidebar.text_input(
        label="Chunk Dataset Repository ID",
        placeholder="Repository ID of the chunk dataset to be indexed",
        value=None,
    )
    chunk_dataset_split = st.sidebar.selectbox(
        label="Chunk Dataset Split",
        options=[
            "pypdf2textloader",
            "pdfplumbertextloader",
            "pymupdf4llmtextloader",
        ],
        index=0,
    )

    index_repo_id = st.sidebar.text_input(
        label="Index Repository ID",
        placeholder="Repository ID of the index artifact to be saved",
    )

    batch_size = st.sidebar.slider(
        label="Batch Size", min_value=4, max_value=1024, value=256, step=4
    )

    start_indexing = st.sidebar.button(label="index")

    if (
        start_indexing
        and not st.session_state.start_indexing
        and chunk_dataset_repo_id is not None
    ):
        st.session_state.start_indexing = True
        if retriever_name != "BM25sRetriever":
            retriever.index(
                chunk_dataset=chunk_dataset_repo_id,
                chunk_dataset_split=chunk_dataset_split,
                index_repo_id=index_repo_id,
                batch_size=batch_size,
                streamlit_mode=True,
            )
        else:
            retriever.index(
                chunk_dataset=chunk_dataset_repo_id,
                chunk_dataset_split=chunk_dataset_split,
                index_repo_id=index_repo_id,
            )
        success_message = "Chunks indexed successfully!"
        if index_repo_id != "":
            success_message += f" Vector index published to https://huggingface.co/datasets/{index_repo_id}"
        st.success(success_message)
        st.balloons()
