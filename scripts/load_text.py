import asyncio
from medrag_multi_modal.document_loader import PDFPlumberTextLoader, PyPDF2TextLoader


URL = "https://archive.org/download/GraysAnatomy41E2015PDF/Gray%27s%20Anatomy%2041E%202015.pdf"
EXCLUDE_PAGES = [
    28,
    29,
    30,
    247,
    330,
    331,
    961,
    1050,
    1051,
    1242,
    1243,
    1244,
    1313,
    1314,
    1445,
    1446,
    1447,
    1448,
]

loader = PDFPlumberTextLoader(
    url=URL,
    document_name="Gray's Anatomy: 41st Edition",
    document_file_path="grays_anatomy.pdf",
    metadata={"source": "https://archive.org/details/GraysAnatomy41E2015PDF"},
)
asyncio.run(
    loader.load_data(
        start_page=20,
        end_page=225,
        exclude_pages=EXCLUDE_PAGES,
        dataset_repo_id="ashwiniai/medrag-text-corpus",
        overwrite_dataset=True,
    )
)

loader = PyPDF2TextLoader(
    url=URL,
    document_name="Gray's Anatomy: 41st Edition",
    document_file_path="grays_anatomy.pdf",
    metadata={"source": "https://archive.org/details/GraysAnatomy41E2015PDF"},
)
asyncio.run(
    loader.load_data(
        start_page=20,
        end_page=225,
        exclude_pages=EXCLUDE_PAGES,
        dataset_repo_id="ashwiniai/medrag-text-corpus",
    )
)