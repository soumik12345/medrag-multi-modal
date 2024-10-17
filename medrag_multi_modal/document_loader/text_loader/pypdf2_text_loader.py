from typing import Dict

import PyPDF2

from .base_text_loader import BaseTextLoader


class PyPDF2TextLoader(BaseTextLoader):
    async def _process_page(self, page_idx: int) -> Dict[str, str]:
        with open(self.document_file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page = pdf_reader.pages[page_idx]
            text = page.extract_text()

        return {
            "text": text,
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
        }
