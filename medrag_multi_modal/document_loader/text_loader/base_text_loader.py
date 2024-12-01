import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import huggingface_hub
import PyPDF2
import streamlit as st
import gdown
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from firerequests import FireRequests
from rich.progress import Progress


class BaseTextLoader(ABC):
    """
    An abstract base class for loading text from a PDF file, processing it into markdown,
    and optionally publishing it to a Weave dataset.

    This class handles the downloading of a PDF file from a given URL if it does not already
    exist locally. Subclasses should implement the specific PDF reading, text extraction,
    and markdown conversion methods.

    The processed pages are finally stored in a list of Page objects, which can be optionally
    published to a Weave dataset.

    Args:
        url (str): The URL of the PDF file to download if not present locally.
        document_name (str): The name of the document for metadata purposes.
        document_file_path (str): The local file path where the PDF is stored or will be downloaded.
        metadata (Optional[dict[str, any]]): Additional metadata to be added to each row of the dataset.
    """

    def __init__(
        self,
        url: str,
        document_name: str,
        document_file_path: str,
        metadata: Optional[dict[str, Any]] = None,
        streamlit_mode: bool = False,
        preview_in_app: bool = False,
    ):
        self.url = url
        self.document_name = document_name
        self.document_file_path = document_file_path
        self.metadata = metadata or {}
        self.streamlit_mode = streamlit_mode
        self.preview_in_app = preview_in_app
        if not os.path.exists(self.document_file_path):
            if self.url.startswith("http"):
                FireRequests().download(self.url, filenames=self.document_file_path)
            else:
                self.url = f"https://drive.google.com/uc?id={self.url}"
                gdown.download(self.url, output=self.document_file_path)

        with open(self.document_file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            self.page_count = len(pdf_reader.pages)

    def get_page_indices(
        self, start_page: Optional[int] = None, end_page: Optional[int] = None
    ) -> tuple[int, int]:
        """
        Get the start and end page indices for processing.

        Args:
            start_page (Optional[int]): The starting page index (0-based) to process.
            end_page (Optional[int]): The ending page index (0-based) to process.

        Returns:
            tuple[int, int]: A tuple containing the start and end page indices.
        """

        if start_page:
            if start_page > self.page_count:
                raise ValueError(
                    f"Start page {start_page} is greater than the total page count {self.page_count}"
                )
        else:
            start_page = 0
        if end_page:
            if end_page > self.page_count:
                raise ValueError(
                    f"End page {end_page} is greater than the total page count {self.page_count}"
                )
        else:
            end_page = self.page_count - 1
        return start_page, end_page

    @abstractmethod
    async def extract_page_data(self, page_idx: int, **kwargs) -> Dict[str, str]:
        """
        Abstract method to process a single page of the PDF and extract the text data.

        Overwrite this method in the subclass to provide the actual implementation and
        processing logic for each page of the PDF using various PDF processing libraries.

        Args:
            page_idx (int): The index of the page to process.
            **kwargs: Additional keyword arguments that may be used by underlying libraries.

        Returns:
            Dict[str, str]: A dictionary containing the processed page data.
        """
        pass

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        exclude_pages: Optional[list[int]] = None,
        dataset_repo_id: Optional[str] = None,
        dataset_split: Optional[str] = None,
        is_dataset_private: bool = False,
        overwrite_dataset: bool = False,
        **kwargs,
    ) -> Dataset:
        """
        Asynchronously loads text from a PDF file specified by a URL or local file path.
        The overrided processing abstract method then processes the text into markdown format,
        and optionally publishes it to a Weave dataset.

        This function downloads a PDF from a given URL if it does not already exist locally,
        reads the specified range of pages, converts each page's content to markdown, and
        returns a list of Page objects containing the text and metadata.

        It uses `PyPDF2` to calculate the number of pages in the PDF and the
        overriden `extract_page_data` method provides the actual implementation to process
        each page, extract the text from the PDF, and convert it to markdown.
        It processes pages concurrently using `asyncio` for efficiency.

        If a `dataset_repo_id` is provided, the processed pages are published to a HuggingFace dataset.

        Args:
            start_page (Optional[int]): The starting page index (0-based) to process.
            end_page (Optional[int]): The ending page index (0-based) to process.
            exclude_pages (Optional[list[int]]): The list of page indices to exclude from processing.
            dataset_repo_id (Optional[str]): The repository ID of the HuggingFace dataset to publish the pages to, if provided.
            dataset_split (Optional[str]): The split of the HuggingFace dataset to publish the pages to, if provided.
            is_dataset_private (bool): Whether the dataset should be private.
            overwrite_dataset (bool): Whether to overwrite the existing dataset if it exists.
            **kwargs: Additional keyword arguments that will be passed to extract_page_data method and the underlying library.

        Returns:
            Dataset: A HuggingFace Dataset object containing the text and metadata for processed pages.
            Each entry in the dataset will have the following keys and values:

            - "text": (str) the processed page data in markdown format.
            - "page_idx": (int) the index of the page.
            - "document_name": (str) the name of the document.
            - "file_path": (str) the local file path where the PDF is stored.
            - "file_url": (str) the URL of the PDF file.
            - "loader_name": (str) the name of the loader class used to process the page.

        Raises:
            ValueError: If the specified start_page or end_page is out of bounds of the document's page count.
        """
        start_page, end_page = self.get_page_indices(start_page, end_page)
        pages = []
        processed_pages_counter: int = 1
        total_pages = end_page - start_page
        exclude_pages = exclude_pages or []
        loader_name = self.__class__.__name__
        dataset_split = loader_name.lower() if dataset_split is None else dataset_split

        if self.preview_in_app and total_pages - len(exclude_pages) > 10:
            warning_message = "Previewing more than 10 pages in app is not recommended due to performance issues."
            if self.streamlit_mode:
                st.warning(warning_message)
            raise ResourceWarning(warning_message)

        streamlit_progressbar = (
            st.progress(
                0,
                text=f"Loading page {processed_pages_counter}/{total_pages} using {loader_name}",
            )
            if self.streamlit_mode
            else None
        )

        async def process_page(page_idx):
            nonlocal processed_pages_counter
            nonlocal streamlit_progressbar
            page_data = await self.extract_page_data(page_idx, **kwargs)
            page_data["loader_name"] = loader_name
            for key, value in self.metadata.items():
                if key not in page_data:
                    page_data[key] = value
            pages.append(page_data)
            progress.update(
                task_id,
                advance=1,
                description=f"Loading page {page_idx} using {loader_name}",
            )
            if streamlit_progressbar:
                progress_percentage = min(
                    100, max(0, int((processed_pages_counter / total_pages) * 100))
                )
                streamlit_progressbar.progress(
                    progress_percentage,
                    text=f"Loading page {page_idx} using {loader_name} ({processed_pages_counter}/{total_pages + 1})",
                )
                if self.preview_in_app:
                    with st.expander(f"Page Index: {page_idx}"):
                        st.markdown(page_data["text"])
            processed_pages_counter += 1

        progress = Progress()
        with progress:
            task_id = progress.add_task("Starting...", total=total_pages)
            tasks = [
                process_page(page_idx)
                for page_idx in range(start_page, end_page + 1)
                if page_idx not in exclude_pages
            ]
            for task in asyncio.as_completed(tasks):
                await task

        pages.sort(key=lambda x: x["page_idx"])

        dataset = Dataset.from_list(pages)
        if dataset_repo_id:
            if huggingface_hub.repo_exists(dataset_repo_id, repo_type="dataset"):
                existing_dataset = load_dataset(dataset_repo_id)
                if not overwrite_dataset:
                    if dataset_split in existing_dataset:
                        existing_dataset[dataset_split] = concatenate_datasets(
                            [dataset, existing_dataset[dataset_split]]
                        )
                        dataset = existing_dataset
                    else:
                        existing_dataset[dataset_split] = dataset
                        dataset = existing_dataset
                else:
                    existing_dataset[dataset_split] = dataset
                    dataset = existing_dataset

            if isinstance(dataset, DatasetDict):
                if "train" in dataset.keys():
                    del dataset["train"]
                dataset.push_to_hub(repo_id=dataset_repo_id, private=is_dataset_private)
            else:
                dataset.push_to_hub(
                    repo_id=dataset_repo_id,
                    private=is_dataset_private,
                    split=dataset_split,
                )

        return dataset
