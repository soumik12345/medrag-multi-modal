import streamlit as st

from medrag_multi_modal.semantic_chunking import SemanticChunker

st.session_state.start_chunking_button = False

st.header("🧇 MedRAG: Chunking")

tokenizer_or_token_counter = st.sidebar.selectbox(
    label="Tokenizer/Token Counter",
    options=[
        "o200k_base",
        "cl100k_base",
        "p50k_base",
        "r50k_base",
    ],
)
chunk_size = st.sidebar.slider("Chunk Size", 64, 64**2, 256, 16)

max_token_chars = st.sidebar.text_input(
    "Max Token Characters (can be left blank by default)", value=None
)
if max_token_chars == "None" or max_token_chars == "":
    max_token_chars = None

memoize = st.sidebar.toggle("Memoize", value=True)

with st.expander("Chunking Job Configuration", expanded=True):
    document_dataset_repo_id = st.text_input(
        "Document Dataset Repository ID", value=None
    )
    chunk_dataset_repo_id = st.text_input("Chunk Dataset Repository ID", value=None)
    dataset_split = st.text_input("Dataset Split", value=None)
    is_dataset_private = st.toggle("Is Dataset Private", value=False)

    st.session_state.start_chunking_button = st.button("Start Chunking")

if st.session_state.start_chunking_button:
    chunker = SemanticChunker(
        tokenizer_or_token_counter=tokenizer_or_token_counter,
        chunk_size=chunk_size,
        max_token_chars=max_token_chars,
        memoize=memoize,
        streamlit_mode=True,
    ).chunk(
        document_dataset_repo_id,
        dataset_split,
        chunk_dataset_repo_id,
        is_dataset_private,
    )
    success_message = "Chunks created successfully!"
    if chunk_dataset_repo_id is not None:
        success_message += f" Dataset published to https://huggingface.co/datasets/{chunk_dataset_repo_id}"
    st.success(success_message)
    st.balloons()
