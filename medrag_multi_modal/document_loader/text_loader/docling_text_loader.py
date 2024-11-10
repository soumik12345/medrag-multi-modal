import os
import shutil
from typing import Any, Optional

from datasets import Dataset
from docling.document_converter import DocumentConverter
from pdf2image.pdf2image import convert_from_path

from medrag_multi_modal.document_loader.text_loader.base_text_loader import (
    BaseTextLoader,
)


class DoclingTextLoader(BaseTextLoader):
    """
    `DoclingTextLoader` is a class designed to handle the extraction and conversion of text
    from PDF documents into a structured format using the docling library. This class extends
    the `BaseTextLoader` and provides additional functionality to convert PDF pages into images
    and subsequently extract text from these images.

    The class utilizes the `pdf2image` library to convert specified pages of a PDF document
    into images. These images are then processed using the `DocumentConverter` from the docling
    library to extract text, which is exported in markdown format.

    !!! example "Example Usage"
        ```python
            import asyncio

            from medrag_multi_modal.document_loader import DoclingTextLoader

            URL = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"

            loader = DoclingTextLoader(
                url=URL,
                document_name="Gray's Anatomy",
                document_file_path="grays_anatomy.pdf",
                image_save_dir="./images",
            )
            dataset = asyncio.run(loader.load_data(start_page=31, end_page=36))
            print(dataset)
        ```

    Attributes:
        url (str): The URL of the PDF document.
        document_name (str): The name of the document.
        document_file_path (str): The path to the PDF file.
        image_save_dir (str): The directory where images of PDF pages are saved.
        metadata (Optional[dict[str, Any]]): Additional metadata related to the document.
    """

    def __init__(
        self,
        url: str,
        document_name: str,
        document_file_path: str,
        image_save_dir: str,
        metadata: Optional[dict[str, Any]] = None,
    ):
        super().__init__(url, document_name, document_file_path, metadata)
        self.image_save_dir = image_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.converter = DocumentConverter()

    async def extract_page_data(self, page_idx: int, **kwargs) -> dict[str, str]:
        """
        Extracts text data from a specific page of the PDF document.

        This function converts a specified page of the PDF document into an image using the
        `pdf2image` library. The image is then saved to the `image_save_dir` directory. The
        saved image is processed using the `DocumentConverter` from the docling library to
        extract text, which is then exported in markdown format.

        Args:
            page_idx (int): The index of the page to be processed (0-based index).
            **kwargs: Additional keyword arguments to be passed to the `convert_from_path` function.

        Returns:
            dict[str, str]: A dictionary containing the extracted text and metadata about the page.
                - "text": The extracted text in markdown format.
                - "page_idx": The index of the processed page.
                - "document_name": The name of the document.
                - "file_path": The path to the PDF file.
                - "file_url": The URL of the PDF document.
        """
        image = convert_from_path(
            self.document_file_path,
            first_page=page_idx + 1,
            last_page=page_idx + 1,
            **kwargs,
        )[0]

        image_file_name = f"page{page_idx}.png"
        image_file_path = os.path.join(self.image_save_dir, image_file_name)
        image.save(image_file_path)

        text = self.converter.convert(image_file_path).document.export_to_markdown()

        return {
            "text": text,
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
        }

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        exclude_pages: Optional[list[int]] = None,
        dataset_repo_id: Optional[str] = None,
        overwrite_dataset: bool = False,
        cleanup: bool = True,
        **kwargs,
    ) -> Dataset:
        """
        Loads data from the PDF document and optionally cleans up the image save directory.

        This function extends the `load_data` method from the superclass to load data from the
        PDF document. It allows specifying a range of pages to load, excluding certain pages,
        and handling dataset repository details. After loading the data, it optionally cleans
        up the image save directory by removing it.

        Args:
            start_page (Optional[int]): The starting page index to load (0-based index).
            end_page (Optional[int]): The ending page index to load (0-based index).
            exclude_pages (Optional[list[int]]): A list of page indices to exclude from loading.
            dataset_repo_id (Optional[str]): The repository ID for the dataset.
            overwrite_dataset (bool): Whether to overwrite the existing dataset.
            cleanup (bool): Whether to clean up the image save directory after loading data.
            **kwargs: Additional keyword arguments to be passed to the superclass `load_data` method.

        Returns:
            Dataset: The loaded dataset containing the extracted data from the PDF document.
        """
        dataset = await super().load_data(
            start_page,
            end_page,
            exclude_pages,
            dataset_repo_id,
            overwrite_dataset,
            **kwargs,
        )

        if cleanup:
            shutil.rmtree(self.image_save_dir)

        return dataset
