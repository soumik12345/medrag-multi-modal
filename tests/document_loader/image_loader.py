import asyncio

from medrag_multi_modal.document_loader.image_loader import (
    FitzPILImageLoader,
    PDF2ImageLoader,
    PDFPlumberImageLoader,
    PyMuPDFImageLoader,
)

URL = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"
COLUMN_NAMES = ["page_image", "page_figure_images", "document_name", "page_idx"]


def test_fitzpil_img_loader():
    loader = FitzPILImageLoader(
        url=URL,
        document_name="Gray's Anatomy",
        document_file_path="grays_anatomy.pdf",
    )
    dataset = asyncio.run(loader.load_data(start_page=32, end_page=37))
    assert dataset.num_rows == 5
    assert dataset.column_names == COLUMN_NAMES
    loader.cleanup_image_dir()


def test_pdf2image_img_loader():
    loader = PDF2ImageLoader(
        url=URL,
        document_name="Gray's Anatomy",
        document_file_path="grays_anatomy.pdf",
    )
    dataset = asyncio.run(loader.load_data(start_page=32, end_page=37))
    assert dataset.num_rows == 5
    assert dataset.column_names == COLUMN_NAMES
    loader.cleanup_image_dir()


def test_pdfplumber_img_loader():
    loader = PDFPlumberImageLoader(
        url=URL,
        document_name="Gray's Anatomy",
        document_file_path="grays_anatomy.pdf",
    )
    dataset = asyncio.run(loader.load_data(start_page=32, end_page=37))
    assert dataset.num_rows == 5
    assert dataset.column_names == COLUMN_NAMES
    loader.cleanup_image_dir()


def test_pymupdf_img_loader():
    loader = PyMuPDFImageLoader(
        url=URL,
        document_name="Gray's Anatomy",
        document_file_path="grays_anatomy.pdf",
    )
    dataset = asyncio.run(loader.load_data(start_page=32, end_page=37))
    assert dataset.num_rows == 5
    assert dataset.column_names == COLUMN_NAMES
    loader.cleanup_image_dir()
