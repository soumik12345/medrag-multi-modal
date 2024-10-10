import asyncio

import rich
import weave
from pdf2image.pdf2image import convert_from_path
from PIL import Image

from medrag_multi_modal.document_loader.load_text import TextLoader


class ImageLoader(TextLoader):
    """
    ImageLoader is a class that extends the TextLoader class to handle the extraction and
    loading of images from a PDF file.

    This class provides functionality to convert specific pages of a PDF document into images
    and optionally publish these images to a Weave dataset.

    !!! example "Example Usage"
        ```python
        import asyncio

        import weave
        from dotenv import load_dotenv

        from medrag_multi_modal.document_loader import ImageLoader

        load_dotenv()
        weave.init(project_name="ml-colabs/medrag-multi-modal")
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
                weave_dataset_name="grays-anatomy-text",
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

    async def load_data(self, start_page: int, end_page: int, weave_dataset_name: str):
        """
        Asynchronously loads images from a PDF file specified by a URL or local file path,
        processes the images for the specified range of pages, and optionally publishes them
        to a Weave dataset.

        This function reads the specified range of pages from a PDF document, converts each page
        to an image using the `pdf2image` library, and returns a list of dictionaries containing
        the image and metadata for each processed page. It processes pages concurrently using
        `asyncio` for efficiency. If a weave_dataset_name is provided, the processed pages are
        published to a Weave dataset.

        Args:
            start_page (int): The starting page index (0-based) to process.
            end_page (int): The ending page index (0-based) to process.
            weave_dataset_name (str): The name of the Weave dataset to publish the pages to,
                if provided.

        Returns:
            list[dict]: A list of dictionaries, each containing the image and metadata for a
                processed page.

        Raises:
            ValueError: If the specified start_page or end_page is out of bounds of the document's
                page count.
        """
        start_page, end_page = self.get_page_indices(start_page, end_page)
        pages = []
        processed_pages_counter: int = 1
        total_pages = end_page - start_page

        async def process_page(page_idx):
            nonlocal processed_pages_counter
            pages.append(
                {
                    "image": convert_from_path(
                        self.document_file_path,
                        first_page=page_idx + 1,
                        last_page=page_idx + 1,
                    )[0],
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
