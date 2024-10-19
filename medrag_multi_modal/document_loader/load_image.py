import asyncio
import os
from typing import Optional

import rich
import weave
from pdf2image.pdf2image import convert_from_path
from PIL import Image

import wandb
from medrag_multi_modal.document_loader.text_loader import PyMuPDF4LLMTextLoader


class ImageLoader(PyMuPDF4LLMTextLoader):
    """
    `ImageLoader` is a class that extends the `TextLoader` class to handle the extraction and
    loading of pages from a PDF file as images.

    This class provides functionality to convert specific pages of a PDF document into images
    and optionally publish these images to a Weave dataset.

    !!! example "Example Usage"
        ```python
        import asyncio

        import wandb
        from dotenv import load_dotenv

        from medrag_multi_modal.document_loader import ImageLoader

        load_dotenv()
        wandb.init(project="medrag-multi-modal", entity="ml-colabs")
        url = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"
        loader = ImageLoader(
            url=url,
            document_name="Gray's Anatomy",
            document_file_path="grays_anatomy.pdf",
        )
        asyncio.run(
            loader.load_data(
                start_page=31,
                end_page=33,
                dataset_name="grays-anatomy-images",
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

    def extract_data_from_pdf_file(
        self, pdf_file: str, page_number: int
    ) -> Image.Image:
        image = convert_from_path(
            pdf_file, first_page=page_number + 1, last_page=page_number + 1
        )[0]
        return image

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        image_save_dir: str = "./images",
        dataset_name: Optional[str] = None,
    ):
        """
        Asynchronously loads images from a PDF file specified by a URL or local file path,
        processes the images for the specified range of pages, and optionally publishes them
        to a Weave dataset.

        This function reads the specified range of pages from a PDF document, converts each page
        to an image using the `pdf2image` library, and returns a list of dictionaries containing
        the image and metadata for each processed page. It processes pages concurrently using
        `asyncio` for efficiency. If a `dataset_name` is provided, the processed page images are
        published to Weights & Biases artifact and the corresponding metadata to a Weave dataset
        with the specified name.

        Args:
            start_page (Optional[int]): The starting page index (0-based) to process.
            end_page (Optional[int]): The ending page index (0-based) to process.
            dataset_name (Optional[str]): The name of the Weave dataset to publish the
                processed images to. Defaults to None.

        Returns:
            list[dict]: A list of dictionaries, each containing the image and metadata for a
                processed page.

        Raises:
            ValueError: If the specified start_page or end_page is out of bounds of the document's
                page count.
        """
        os.makedirs(image_save_dir, exist_ok=True)
        start_page, end_page = self.get_page_indices(start_page, end_page)
        pages = []
        processed_pages_counter: int = 1
        total_pages = end_page - start_page

        async def process_page(page_idx):
            nonlocal processed_pages_counter
            image = convert_from_path(
                self.document_file_path,
                first_page=page_idx + 1,
                last_page=page_idx + 1,
            )[0]
            pages.append(
                {
                    "page_idx": page_idx,
                    "document_name": self.document_name,
                    "file_path": self.document_file_path,
                    "file_url": self.url,
                }
            )
            image.save(os.path.join(image_save_dir, f"{page_idx}.png"))
            rich.print(f"Processed pages {processed_pages_counter}/{total_pages}")
            processed_pages_counter += 1

        tasks = [process_page(page_idx) for page_idx in range(start_page, end_page)]
        for task in asyncio.as_completed(tasks):
            await task
        if dataset_name:
            artifact = wandb.Artifact(name=dataset_name, type="dataset")
            artifact.add_dir(local_path=image_save_dir)
            artifact.save()
            weave.publish(weave.Dataset(name=dataset_name, rows=pages))
        return pages
