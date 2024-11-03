from typing import Dict

import pymupdf4llm

from medrag_multi_modal.document_loader.text_loader.base_text_loader import (
    BaseTextLoader,
)


class PyMuPDF4LLMTextLoader(BaseTextLoader):
    """
    A concrete implementation of the BaseTextLoader for loading text from a PDF file,
    processing it into markdown using `pymupdf4llm`, and optionally publishing it to a Weave dataset.

    This class extends the BaseTextLoader and implements the abstract methods to load and process pages from a PDF file.

    This class will handle the downloading of a PDF file from a given URL if it does not already exist locally.
    It uses PyPDF2 to read the PDF and pymupdf4llm to convert pages to markdown. The processed pages are stored in a list
    of Page objects, which can be optionally published to a Weave dataset.

    !!! example "Example Usage"
        ```python
        import asyncio
        
        from medrag_multi_modal.document_loader import PyMuPDF4LLMTextLoader

        URL = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"

        loader = PyMuPDF4LLMTextLoader(
            url=URL,
            document_name="Gray's Anatomy",
            document_file_path="grays_anatomy.pdf",
        )
        dataset = asyncio.run(loader.load_data(start_page=31, end_page=36))
        ```

    Args:
        url (str): The URL of the PDF file to download if not present locally.
        document_name (str): The name of the document for metadata purposes.
        document_file_path (str): The local file path where the PDF is stored or will be downloaded.
    """

    async def extract_page_data(self, page_idx: int, **kwargs) -> Dict[str, str]:
        """
        Process a single page of the PDF and convert it to markdown using `pymupdf4llm`.

        Returns:
            Dict[str, str]: A dictionary with the processed page data.
            The dictionary will have the following keys and values:

            - "text": (str) the processed page data in markdown format.
            - "page_idx": (int) the index of the page.
            - "document_name": (str) the name of the document.
            - "file_path": (str) the local file path where the PDF is stored.
            - "file_url": (str) the URL of the PDF file.

        Args:
            page_idx (int): The index of the page to process.
            **kwargs: Additional keyword arguments to be passed to `pymupdf4llm.to_markdown`.

        Returns:
            Dict[str, str]: A dictionary containing the processed page data.
        """
        text = pymupdf4llm.to_markdown(
            doc=self.document_file_path, pages=[page_idx], show_progress=False, **kwargs
        )
        return {
            "text": text,
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
        }
