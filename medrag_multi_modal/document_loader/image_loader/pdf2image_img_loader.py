import os
from typing import Any, Dict

from pdf2image.pdf2image import convert_from_path

from .base_img_loader import BaseImageLoader


class PDF2ImageLoader(BaseImageLoader):
    """
    `PDF2ImageLoader` is a class that extends the `BaseImageLoader` class to handle the extraction and
    loading of pages from a PDF file as images using the pdf2image library.

    This class provides functionality to convert specific pages of a PDF document into images
    and optionally publish these images to a Weave artifact.
    It is like a snapshot image version of each of the pages from the PDF.

    Args:
        url (str): The URL of the PDF document.
        document_name (str): The name of the document.
        document_file_path (str): The path to the PDF file.
    """

    def __init__(self, url: str, document_name: str, document_file_path: str):
        super().__init__(url, document_name, document_file_path)

    async def extract_page_data(
        self, page_idx: int, image_save_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Extracts a single page from the PDF as an image using pdf2image library.

        Args:
            page_idx (int): The index of the page to process.
            image_save_dir (str): The directory to save the extracted image.
            **kwargs: Additional keyword arguments that may be used by pdf2image.

        Returns:
            Dict[str, Any]: A dictionary containing the processed page data.
            The dictionary will have the following keys and values:

            - "page_idx": (int) the index of the page.
            - "document_name": (str) the name of the document.
            - "file_path": (str) the local file path where the PDF is stored.
            - "file_url": (str) the URL of the PDF file.
            - "image_file_path": (str) the local file path where the image is stored.
        """
        image = convert_from_path(
            self.document_file_path,
            first_page=page_idx + 1,
            last_page=page_idx + 1,
            **kwargs,
        )[0]

        image_file_name = f"page{page_idx}.png"
        image_file_path = os.path.join(image_save_dir, image_file_name)
        image.save(image_file_path)

        return {
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
            "image_file_path": image_file_path,
        }
