from typing import Dict

from marker.convert import convert_single_pdf
from marker.models import load_all_models

from .base_text_loader import BaseTextLoader


class MarkerTextLoader(BaseTextLoader):
    """
    A concrete implementation of the BaseTextLoader for loading text from a PDF file
    using `marker-pdf`, processing it into a structured text format, and optionally publishing
    it to a Weave dataset.

    This class extends the BaseTextLoader and implements the abstract methods to
    load and process pages from a PDF file using marker-pdf, which is a pipeline of deep learning models.

    This class will handle the downloading of a PDF file from a given URL if it does not already exist locally.
    It uses marker-pdf to read the PDF and extract structured text from each page. The processed pages are stored
    in a list of Page objects, which can be optionally published to a Weave dataset.

    !!! example "Example Usage"
        ```python
        import asyncio

        import weave

        from medrag_multi_modal.document_loader.text_loader import MarkerTextLoader

        weave.init(project_name="ml-colabs/medrag-multi-modal")
        url = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"
        loader = MarkerTextLoader(
            url=url,
            document_name="Gray's Anatomy",
            document_file_path="grays_anatomy.pdf",
        )
        asyncio.run(
            loader.load_data(
                start_page=31,
                end_page=36,
                weave_dataset_name="grays-anatomy-text",
            )
        )
        ```

    Args:
        url (str): The URL of the PDF file to download if not present locally.
        document_name (str): The name of the document for metadata purposes.
        document_file_path (str): The local file path where the PDF is stored or will be downloaded.
    """

    async def _process_page(self, page_idx: int) -> Dict[str, str]:
        """
        Process a single page of the PDF and extract its structured text using marker-pdf.

        Returns a dictionary with the processed page data.
        The dictionary will have the following keys and values:
            - "text": (str) the extracted structured text from the page.
            - "page_idx": (int) the index of the page.
            - "document_name": (str) the name of the document.
            - "file_path": (str) the local file path where the PDF is stored.
            - "file_url": (str) the URL of the PDF file.
            - "meta": (dict) the metadata extracted from the page by marker-pdf.

        Args:
            page_idx (int): The index of the page to process.

        Returns:
            Dict[str, str]: A dictionary containing the processed page data.
        """
        model_lst = load_all_models()

        text, _, out_meta = convert_single_pdf(
            self.document_file_path,
            model_lst,
            max_pages=1,
            batch_multiplier=1,
            start_page=page_idx,
            ocr_all_pages=True,
        )

        return {
            "text": text,
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
            "meta": out_meta,
        }
