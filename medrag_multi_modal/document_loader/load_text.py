import asyncio
import os
from typing import Optional

import pymupdf4llm
import PyPDF2
import rich
import weave
from firerequests import FireRequests


class TextLoader:
    """
    A class for loading text from a PDF file, processing it into markdown, and optionally publishing it to a Weave dataset.

    This class handles the downloading of a PDF file from a given URL if it does not already exist locally.
    It uses PyPDF2 to read the PDF and pymupdf4llm to convert pages to markdown. The processed pages are stored in a list
    of Page objects, which can be optionally published to a Weave dataset.

    !!! example "Example Usage"
        ```python
        import asyncio

        import weave

        from medrag_multi_modal.document_loader import TextLoader

        weave.init(project_name="ml-colabs/medrag-multi-modal")
        url = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"
        loader = TextLoader(
            url=url,
            document_name="Gray's Anatomy",
            document_file_path="grays_anatomy.pdf",
        )
        asyncio.run(
            loader.load_data(start_page=9, end_page=15, weave_dataset_name="grays-anatomy-text")
        )
        ```

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
    ):
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

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        weave_dataset_name: Optional[str] = None,
    ):
        """
        Asynchronously loads text from a PDF file specified by a URL or local file path,
        processes the text into markdown format, and optionally publishes it to a Weave dataset.

        This function downloads a PDF from a given URL if it does not already exist locally,
        reads the specified range of pages, converts each page's content to markdown, and
        returns a list of Page objects containing the text and metadata. It uses PyPDF2 to read
        the PDF and pymupdf4llm to convert pages to markdown. It processes pages concurrently using
        `asyncio` for efficiency. If a weave_dataset_name is provided, the processed pages are published
        to a Weave dataset.

        Args:
            start_page (Optional[int]): The starting page index (0-based) to process. Defaults to the first page.
            end_page (Optional[int]): The ending page index (0-based) to process. Defaults to the last page.
            weave_dataset_name (Optional[str]): The name of the Weave dataset to publish the pages to, if provided.

        Returns:
            list[Page]: A list of Page objects, each containing the text and metadata for a processed page.

        Raises:
            ValueError: If the specified start_page or end_page is out of bounds of the document's page count.
        """
        start_page, end_page = self.get_page_indices(start_page, end_page)
        pages = []
        processed_pages_counter: int = 1
        total_pages = end_page - start_page

        async def process_page(page_idx):
            nonlocal processed_pages_counter
            text = pymupdf4llm.to_markdown(
                doc=self.document_file_path, pages=[page_idx], show_progress=False
            )
            pages.append(
                {
                    "text": text,
                    "page_idx": page_idx,
                    "document_name": self.document_name,
                    "file_path": self.document_file_path,
                    "file_url": self.url,
                }
            )
            rich.print(f"Processed pages {processed_pages_counter}/{total_pages}")
            processed_pages_counter += 1

        tasks = [process_page(page_idx) for page_idx in range(start_page, end_page)]
        for task in asyncio.as_completed(tasks):
            await task
        if weave_dataset_name:
            weave.publish(weave.Dataset(name=weave_dataset_name, rows=pages))
        return pages
