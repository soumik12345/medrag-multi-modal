from typing import Dict

import PyPDF2

from medrag_multi_modal.document_loader.text_loader.base_text_loader import (
    BaseTextLoader,
)


class PyPDF2TextLoader(BaseTextLoader):
    """
    A concrete implementation of the BaseTextLoader for loading text from a PDF file
    using `PyPDF2`, processing it into a simple text format, and optionally publishing
    it to a Weave dataset.

    This class extends the BaseTextLoader and implements the abstract methods to
    load and process pages from a PDF file.

    This class will handle the downloading of a PDF file from a given URL if it does not already exist locally.
    It uses PyPDF2 to read the PDF and extract text from each page. The processed pages are stored in a list
    of Page objects, which can be optionally published to a Weave dataset.

    !!! example "Example Usage"
        ```python
        import asyncio
        
        from medrag_multi_modal.document_loader import PyPDF2TextLoader

        URL = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"

        loader = PyPDF2TextLoader(
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
        Process a single page of the PDF and extract its text using PyPDF2.

        Returns:
            Dict[str, str]: A dictionary with the processed page data.
            The dictionary will have the following keys and values:

            - "text": (str) the extracted text from the page.
            - "page_idx": (int) the index of the page.
            - "document_name": (str) the name of the document.
            - "file_path": (str) the local file path where the PDF is stored.
            - "file_url": (str) the URL of the PDF file.

        Args:
            page_idx (int): The index of the page to process.
            **kwargs: Additional keyword arguments to be passed to `PyPDF2.PdfReader.pages[0].extract_text`.

        Returns:
            Dict[str, str]: A dictionary containing the processed page data.
        """
        with open(self.document_file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page = pdf_reader.pages[page_idx]
            text = page.extract_text(**kwargs)

        return {
            "text": text,
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
        }
