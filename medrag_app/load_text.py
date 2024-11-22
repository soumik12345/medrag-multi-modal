import asyncio
import importlib
import os

import streamlit as st
from firerequests import FireRequests

st.header("📘 MedRAG: Text Loading")

document_file_path = "temp.pdf"
document_url = None

# Flag to control widget visibility
proceed_to_text_loader_configuration = False

st.session_state.load_text = False
if os.path.exists(document_file_path):
    proceed_to_text_loader_configuration = st.session_state.document_downloaded = True
elif "document_downloaded" not in st.session_state:
    st.session_state.document_downloaded = False
    st.session_state.load_text = False

if not st.session_state.document_downloaded:
    document_url = st.sidebar.text_input(
        label="Document URL",
        placeholder="URL of the document to load",
    )

    download_button = st.sidebar.button("Download Document")
    if download_button and not st.session_state.document_downloaded:
        with st.sidebar.status("Downloading document..."):
            try:
                FireRequests().download(document_url, filenames="temp.pdf")
                proceed_to_text_loader_configuration = True  # Set flag after successful download
            except Exception as e:
                st.sidebar.error(f"Download failed: {e}")
        st.session_state.document_downloaded = True


if proceed_to_text_loader_configuration:  # Only show widgets if download is complete or file exists
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
            start_page, end_page = st.select_slider(
                label="Pages",
                options=list(range(1, text_loader.page_count + 1)),
                value=(1, text_loader.page_count),
            )

            load_text_button = st.sidebar.button("Load Text")
            if load_text_button and not st.session_state.load_text:
                st.session_state.load_text = True
                with st.spinner("Loading text..."):
                    dataset = asyncio.run(
                        text_loader.load_data(start_page=start_page, end_page=end_page)
                    )
                st.success("Text loaded successfully!")
