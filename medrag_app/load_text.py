import asyncio
import importlib
import os

import streamlit as st
from datasets import Dataset
from firerequests import FireRequests

st.header("📘 MedRAG: Text Loading")

document_file_path = "temp.pdf"
document_url = None

proceed_to_text_loader_configuration = False
dataset: Dataset | None = None

st.session_state.load_text = False
if os.path.exists(document_file_path):
    proceed_to_text_loader_configuration = st.session_state.document_downloaded = True
elif "document_downloaded" not in st.session_state:
    st.session_state.document_downloaded = False
    st.session_state.load_text = False


document_url = st.sidebar.text_input(
    label="Document URL",
    placeholder="URL of the document to load",
)

if not st.session_state.document_downloaded:
    download_button = st.sidebar.button("Download Document")
    if download_button and not st.session_state.document_downloaded:
        with st.sidebar.status("Downloading document..."):
            try:
                FireRequests().download(document_url, filenames="temp.pdf")
                proceed_to_text_loader_configuration = True
            except Exception as e:
                st.sidebar.error(f"Download failed: {e}")
        st.session_state.document_downloaded = True


if proceed_to_text_loader_configuration:
    document_name = st.sidebar.text_input(
        label="Document Name",
        placeholder="Name of the document to load",
    )
    text_loader_name = st.sidebar.selectbox(
        label="Text Loader",
        options=[
            "",
            "MarkerTextLoader",
            "PDFPlumberTextLoader",
            "PyMuPDF4LLMTextLoader",
            "PyPDF2TextLoader",
        ],
    )
    if text_loader_name != "":
        module_name = "medrag_multi_modal.document_loader.text_loader"
        text_loader_class = getattr(
            importlib.import_module(module_name), text_loader_name
        )

        preview_in_app = st.sidebar.toggle("Preview in App")

        text_loader = text_loader_class(
            url=document_url,
            document_name=document_name,
            document_file_path=document_file_path,
            streamlit_mode=True,
            preview_in_app=preview_in_app,
        )

        if text_loader.page_count > 0:
            with st.expander(f"{text_loader_name} Configuration"):
                start_page, end_page = st.select_slider(
                    label="Pages",
                    options=list(range(1, text_loader.page_count + 1)),
                    value=(1, text_loader.page_count),
                )

                dataset_repo_id = st.text_input(
                    label="HuggingFace Dataset Repository",
                    placeholder="Repository ID of the dataset to publish the pages to",
                )

                is_dataset_private = st.toggle("Private Dataset")

                load_text_button = st.button("Load Text")

            if load_text_button and not st.session_state.load_text:
                st.session_state.load_text = True
                with st.spinner("Loading text..."):
                    dataset = asyncio.run(
                        text_loader.load_data(
                            start_page=start_page,
                            end_page=end_page,
                            dataset_repo_id=dataset_repo_id,
                            is_dataset_private=is_dataset_private,
                        )
                    )
                success_message = "Text loaded successfully!"
                if dataset_repo_id != "":
                    success_message += f" Dataset published to https://huggingface.co/datasets/{dataset_repo_id}"
                st.success(success_message)
                st.balloons()

            if dataset is not None and preview_in_app:
                st.dataframe(data=dataset.to_pandas())
