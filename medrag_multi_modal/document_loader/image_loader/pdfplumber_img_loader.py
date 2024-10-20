import os
from typing import Any, Dict

import pdfplumber

from .base_img_loader import BaseImageLoader


class PDFPlumberImageLoader(BaseImageLoader):
    """
    `PDFPlumberImageLoader` is a class that extends the `BaseImageLoader` class to handle the extraction and
    loading of pages from a PDF file as images using the pdfplumber library.

    This class provides functionality to extract images from a PDF file using pdfplumber library,
    and optionally publish these images to a WandB artifact.

    !!! example "Example Usage"
        ```python
        import asyncio

        import weave

        import wandb
        from medrag_multi_modal.document_loader.image_loader import PDFPlumberImageLoader

        weave.init(project_name="ml-colabs/medrag-multi-modal")
        wandb.init(project="medrag-multi-modal", entity="ml-colabs")
        url = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"
        loader = PDFPlumberImageLoader(
            url=url,
            document_name="Gray's Anatomy",
            document_file_path="grays_anatomy.pdf",
        )
        asyncio.run(
            loader.load_data(
                start_page=32,
                end_page=37,
                wandb_artifact_name="grays-anatomy-images-pdfplumber",
                cleanup=False,
            )
        )
        ```

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
        Extracts a single page from the PDF as an image using pdfplumber library.

        Args:
            page_idx (int): The index of the page to process.
            image_save_dir (str): The directory to save the extracted image.
            **kwargs: Additional keyword arguments that may be used by pdfplumber.

        Returns:
            Dict[str, Any]: A dictionary containing the processed page data.
            The dictionary will have the following keys and values:

            - "page_idx": (int) the index of the page.
            - "document_name": (str) the name of the document.
            - "file_path": (str) the local file path where the PDF is stored.
            - "file_url": (str) the URL of the PDF file.
            - "image_file_path": (str) the local file path where the image is stored.
        """
        with pdfplumber.open(self.document_file_path) as pdf:
            page = pdf.pages[page_idx]
            images = page.images

            image_file_paths = []
            for img_idx, image in enumerate(images):
                extracted_image = page.crop(
                    (
                        image["x0"],
                        image["top"],
                        image["x1"],
                        image["bottom"],
                    )
                ).to_image(resolution=300)

                image_file_name = f"page{page_idx}_fig{img_idx}.png"
                image_file_path = os.path.join(image_save_dir, image_file_name)

                extracted_image.save(image_file_path, "png")
                image_file_paths.append(image_file_path)

        return {
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
            "image_file_paths": image_file_paths,
        }
