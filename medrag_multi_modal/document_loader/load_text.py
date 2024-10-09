import asyncio
import os
from typing import Optional

import pymupdf4llm
import PyPDF2
import rich
import weave
from firerequests import FireRequests
from pydantic import BaseModel


class Page(BaseModel):
    text: str
    page_idx: int
    document_name: str
    file_path: str
    file_url: str


async def load_text_from_pdf(
    url: str,
    document_name: str,
    document_file_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    weave_dataset_name: Optional[str] = None,
) -> list[Page]:
    """
    Asynchronously loads text from a PDF file specified by a URL or local file path,
    processes the text into markdown format, and optionally publishes it to a Weave dataset.

    This function downloads a PDF from a given URL if it does not already exist locally,
    reads the specified range of pages, converts each page's content to markdown, and
    returns a list of Page objects containing the text and metadata. It uses PyPDF2 to read
    the PDF and pymupdf4llm to convert pages to markdown. It processes pages concurrently using
    `asyncio` for efficiency. If a weave_dataset_name is provided, the processed pages are published
    to a Weave dataset.

    !!! example "Example usage"
        ```python
        import asyncio

        import weave

        from medrag_multi_modal.document_loader import load_text_from_pdf

        weave.init(project_name="ml-colabs/medrag-multi-modal")
        url = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"
        asyncio.run(
            load_text_from_pdf(
                url=url,
                document_name="Gray's Anatomy",
                start_page=9,
                end_page=15,
                document_file_path="grays_anatomy.pdf",
            )
        )
        ```

    Args:
        url (str): The URL of the PDF file to download if not present locally.
        document_name (str): The name of the document for metadata purposes.
        document_file_path (str): The local file path where the PDF is stored or will be downloaded.
        start_page (Optional[int]): The starting page index (0-based) to process. Defaults to the first page.
        end_page (Optional[int]): The ending page index (0-based) to process. Defaults to the last page.
        weave_dataset_name (Optional[str]): The name of the Weave dataset to publish the pages to, if provided.

    Returns:
        list[Page]: A list of Page objects, each containing the text and metadata for a processed page.

    Raises:
        ValueError: If the specified start_page or end_page is out of bounds of the document's page count.
    """
    if not os.path.exists(document_file_path):
        FireRequests().download(url, filename=document_file_path)
    with open(document_file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        page_count = len(pdf_reader.pages)
    print(f"Page count: {page_count}")
    if start_page:
        if start_page > page_count:
            raise ValueError(
                f"Start page {start_page} is greater than the total page count {page_count}"
            )
    else:
        start_page = 0
    if end_page:
        if end_page > page_count:
            raise ValueError(
                f"End page {end_page} is greater than the total page count {page_count}"
            )
    else:
        end_page = page_count - 1

    pages: list[Page] = []
    processed_pages_counter: int = 1
    total_pages = end_page - start_page

    async def process_page(page_idx):
        nonlocal processed_pages_counter
        text = pymupdf4llm.to_markdown(
            doc=document_file_path, pages=[page_idx], show_progress=False
        )
        pages.append(
            Page(
                text=text,
                page_idx=page_idx,
                document_name=document_name,
                file_path=document_file_path,
                file_url=url,
            )
        )
        rich.print(f"Processed pages {processed_pages_counter}/{total_pages}")
        processed_pages_counter += 1

    tasks = [process_page(page_idx) for page_idx in range(start_page, end_page)]
    for task in asyncio.as_completed(tasks):
        await task
    if weave_dataset_name:
        weave.publish(weave.Dataset(name=weave_dataset_name, rows=pages))
    return pages
