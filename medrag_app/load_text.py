import asyncio
import importlib
import os

import streamlit as st
import gdown
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
    placeholder="URL or Google Drive ID",
    help="URL or Google Drive ID of the document to load, in the format https://drive.google.com/uc?id=<ID> or full URL to the document",
)

if not st.session_state.document_downloaded:
    download_button = st.sidebar.button("Download Document")
    if download_button and not st.session_state.document_downloaded:
        with st.sidebar.status("Downloading document..."):
            try:
                if document_url.startswith("http"):
                    FireRequests().download(document_url, filenames="temp.pdf")
                else:
                    document_url = f"https://drive.google.com/uc?id={document_url}"
                    gdown.download(document_url, output="temp.pdf")

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
            with st.expander(f"{text_loader_name} Configuration", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    start_page = st.number_input(
                        "Start Page Index (min: 0)",
                        min_value=0,
                        max_value=text_loader.page_count - 1,
                        value=0,
                    )
                with col2:
                    end_page = st.number_input(
                        f"End Page Index (max: {text_loader.page_count-1})",
                        min_value=start_page,
                        max_value=text_loader.page_count - 1,
                        value=text_loader.page_count - 1,
                    )

                if "excluded_pages" not in st.session_state:
                    st.session_state.excluded_pages = []

                excluded_pages = st.multiselect(
                    label="Excluded Pages",
                    options=list(range(start_page, end_page + 1)),
                    default=st.session_state.excluded_pages,
                    placeholder="Select pages to exclude",
                    key="excluded_pages_multiselect",
                )

                if (
                    "excluded_pages_multiselect" in st.session_state
                    and st.session_state.excluded_pages_multiselect
                    != st.session_state.excluded_pages
                ):
                    st.session_state.excluded_pages = (
                        st.session_state.excluded_pages_multiselect.copy()
                    )
                    st.rerun()

                @st.dialog("Bulk Exclude Pages")
                def bulk_exclude_pages():
                    csv_input = st.text_area(
                        "Paste comma-separated page numbers to exclude",
                        placeholder="example: 1, 2, 3, 4, 5",
                    )

                    if st.button("Exclude Pages"):
                        if csv_input:
                            try:
                                pages = [int(x.strip()) for x in csv_input.split(",")]

                                out_of_range = [
                                    p for p in pages if p < start_page or p > end_page
                                ]
                                if out_of_range:
                                    st.error(
                                        f"Pages {sorted(out_of_range)} are outside the valid range ({start_page}-{end_page})"
                                    )
                                    return

                                already_excluded = [
                                    p
                                    for p in pages
                                    if p in st.session_state.excluded_pages
                                ]
                                if already_excluded:
                                    st.warning(
                                        f"Pages {already_excluded} are already in the exclusion list"
                                    )

                                valid_pages = [
                                    p
                                    for p in pages
                                    if start_page <= p <= end_page
                                    and p not in st.session_state.excluded_pages
                                ]

                                if valid_pages:
                                    st.session_state.excluded_pages = sorted(
                                        st.session_state.excluded_pages + valid_pages
                                    )
                                    st.success(
                                        f"Added {len(valid_pages)} pages to exclusion list"
                                    )
                                    st.rerun()
                                else:
                                    st.warning("No new valid pages found to exclude")

                            except ValueError:
                                st.error(
                                    "Invalid input format. Please enter comma-separated numbers (example: 1, 2, 3)."
                                )

                if st.button("📋", help="Bulk exclude pages"):
                    bulk_exclude_pages()

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
                            exclude_pages=(
                                [int(i) for i in excluded_pages]
                                if excluded_pages
                                else None
                            ),
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
