import asyncio

from medrag_multi_modal.document_loader import (
    PDFPlumberTextLoader,
    PyMuPDF4LLMTextLoader,
    PyPDF2TextLoader,
)

URL = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"
COLUMN_NAMES = [
    "text",
    "page_idx",
    "document_name",
    "file_path",
    "file_url",
    "loader_name",
]


def test_pdfplumber_text_loader():
    loader = PDFPlumberTextLoader(
        url=URL,
        document_name="Gray's Anatomy",
        document_file_path="grays_anatomy.pdf",
    )
    dataset = asyncio.run(loader.load_data(start_page=31, end_page=36))
    assert dataset.num_rows == 6
    assert dataset.column_names == COLUMN_NAMES


def test_pymupdf_text_loader():
    loader = PyMuPDF4LLMTextLoader(
        url=URL,
        document_name="Gray's Anatomy",
        document_file_path="grays_anatomy.pdf",
    )
    dataset = asyncio.run(loader.load_data(start_page=31, end_page=36))
    assert dataset.num_rows == 6
    assert dataset.column_names == COLUMN_NAMES


def test_pypdf2_text_loader():
    loader = PyPDF2TextLoader(
        url=URL,
        document_name="Gray's Anatomy",
        document_file_path="grays_anatomy.pdf",
    )
    dataset = asyncio.run(loader.load_data(start_page=31, end_page=36))
    assert dataset.num_rows == 6
    assert dataset.column_names == COLUMN_NAMES
