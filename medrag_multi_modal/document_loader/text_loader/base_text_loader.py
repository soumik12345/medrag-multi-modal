import asyncio
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import PyPDF2
import rich
import weave
from firerequests import FireRequests


class BaseTextLoader(ABC):
    """
    An abstract base class for loading text from a PDF file, processing it into markdown, and optionally publishing it to a Weave dataset.

    This class handles the downloading of a PDF file from a given URL if it does not already exist locally.
    Subclasses should implement the specific PDF reading, text extraction, and markdown conversion methods.

    The processed pages are finally stored in a list of Page objects, which can be optionally published to a Weave dataset.

    Args:
        url (str): The URL of the PDF file to download if not present locally.
        document_name (str): The name of the document for metadata purposes.
        document_file_path (str): The local file path where the PDF is stored or will be downloaded.
    """

    def __init__(self, url: str, document_name: str, document_file_path: str):
        self.url = url
        self.document_name = document_name
        self.document_file_path = document_file_path
        if not os.path.exists(self.document_file_path):
            FireRequests().download(url, filename=self.document_file_path)
        with open(self.document_file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            self.page_count = len(pdf_reader.pages)

    def get_page_indices(
        self, start_page: Optional[int] = None, end_page: Optional[int] = None
    ) -> tuple[int, int]:
        """
        Get the start and end page indices for processing.

        Args:
            start_page (Optional[int]): The starting page index (0-based) to process. Defaults to the first page.
            end_page (Optional[int]): The ending page index (0-based) to process. Defaults to the last page.

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
    async def extract_page_data(self, page_idx: int) -> Dict[str, str]:
        """
        Abstract method to process a single page of the PDF and extract the text data.

        Overwrite this method in the subclass to provide the actual implementation and
        processing logic for each page of the PDF using various PDF processing libraries.

        Args:
            page_idx (int): The index of the page to process.

        Returns:
            Dict[str, str]: A dictionary containing the processed page data.
        """
        pass

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        weave_dataset_name: Optional[str] = None,
    ) -> List[Dict[str, str]]:
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

        If a weave_dataset_name is provided, the processed pages are published to a Weave dataset.

        Args:
            start_page (Optional[int]): The starting page index (0-based) to process. Defaults to the first page.
            end_page (Optional[int]): The ending page index (0-based) to process. Defaults to the last page.
            weave_dataset_name (Optional[str]): The name of the Weave dataset to publish the pages to, if provided.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing the text and metadata for a processed page.
            Each dictionary will have the following keys and values:

            - "text": (str) the processed page data in markdown format.
            - "page_idx": (int) the index of the page.
            - "document_name": (str) the name of the document.
            - "file_path": (str) the local file path where the PDF is stored.
            - "file_url": (str) the URL of the PDF file.

        Raises:
            ValueError: If the specified start_page or end_page is out of bounds of the document's page count.
        """
        start_page, end_page = self.get_page_indices(start_page, end_page)
        pages = []
        processed_pages_counter: int = 1
        total_pages = end_page - start_page

        async def process_page(page_idx):
            nonlocal processed_pages_counter
            page_data = await self.extract_page_data(page_idx)
            pages.append(page_data)
            rich.print(
                f"Processed page idx: {page_idx}, progress: {processed_pages_counter}/{total_pages}"
            )
            processed_pages_counter += 1

        tasks = [process_page(page_idx) for page_idx in range(start_page, end_page)]
        for task in asyncio.as_completed(tasks):
            await task

        if weave_dataset_name:
            weave.publish(weave.Dataset(name=weave_dataset_name, rows=pages))
        return pages
