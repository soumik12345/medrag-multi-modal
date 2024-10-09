import asyncio
import os
from glob import glob
from typing import Optional

import pymupdf4llm
import rich
import weave
from PIL import Image

from medrag_multi_modal.document_loader.load_text import TextLoader


class TextImageLoader(TextLoader):
    """
    A class for loading and processing text and images from a document.

    The TextImageLoader class extends the TextLoader class to provide
    functionality for extracting both text and images from a document
    specified by a URL, document name, and file path. It processes the
    document asynchronously, allowing for efficient handling of large
    documents.

    !!! example "Example Usage"
        ```python
        import asyncio

        import weave

        from medrag_multi_modal.document_loader import TextImageLoader

        weave.init(project_name="ml-colabs/medrag-multi-modal")
        url = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"
        loader = TextImageLoader(
            url=url,
            document_name="Gray's Anatomy",
            document_file_path="grays_anatomy.pdf",
        )
        asyncio.run(
            loader.load_data(
                start_page=20,
                end_page=25,
                weave_dataset_name="grays-anatomy-text",
            )
        )
        ```

    Args:
        url (str): The URL of the document to be processed.
        document_name (str): The name of the document.
        document_file_path (str): The file path where the document is stored.
    """

    def __init__(self, url: str, document_name: str, document_file_path: str):
        super().__init__(url, document_name, document_file_path)

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        weave_dataset_name: Optional[str] = None,
        image_path: Optional[str] = "./images",
        dpi: int = 300,
    ):
        """
        Asynchronously loads and processes text and images from a specified range of pages
        in a document. This function extracts text in markdown format and images in PNG
        format from the document, storing them in a list of dictionaries, each representing
        a page. Optionally, the processed data can be published to a Weave dataset.

        The function first determines the page indices to process using the
        `get_page_indices` method. It then defines an asynchronous inner function,
        `process_page`, which handles the extraction of text and images for a single page.
        The text is extracted using the `pymupdf4llm.to_markdown` function, and images are
        retrieved from the specified image path. The processed data is appended to the
        `pages` list.

        The function creates a list of tasks for processing each page asynchronously and
        awaits their completion. If a `weave_dataset_name` is provided, the processed data
        is published to a Weave dataset. Finally, the function returns the list of processed
        pages.

        Args:
            start_page (Optional[int]): The starting page index for processing. If None,
                defaults to the first page of the document.
            end_page (Optional[int]): The ending page index for processing. If None,
                defaults to the last page of the document.
            weave_dataset_name (Optional[str]): The name of the Weave dataset to publish
                the processed data to. If None, the data is not published.
            image_path (Optional[str]): The directory path where extracted images are
                stored. Defaults to "./images".
            dpi (int): The resolution in dots per inch for image extraction. Defaults to 300.

        Returns:
            List[Dict]: A list of dictionaries, each containing the extracted text, page
            index, document name, file path, file URL, and a list of images for each page
            processed.
        """
        start_page, end_page = self.get_page_indices(start_page, end_page)
        pages = []
        processed_pages_counter: int = 1
        total_pages = end_page - start_page

        async def process_page(page_idx):
            nonlocal processed_pages_counter
            text = pymupdf4llm.to_markdown(
                doc=self.document_file_path,
                pages=[page_idx],
                show_progress=False,
                write_images=True,
                image_format="png",
                dpi=dpi,
                image_path=image_path,
            )
            image_paths = glob(
                os.path.join(image_path, f"{self.document_file_path}-{page_idx}-*.png")
            )
            print(image_paths)
            pages.append(
                {
                    "text": text,
                    "page_idx": page_idx,
                    "document_name": self.document_name,
                    "file_path": self.document_file_path,
                    "file_url": self.url,
                    "images": [Image.open(image) for image in image_paths],
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
