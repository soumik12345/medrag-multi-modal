from typing import Dict

import pdfplumber

from .base_text_loader import BaseTextLoader


class PDFPlumberTextLoader(BaseTextLoader):
    async def _process_page(self, page_idx: int) -> Dict[str, str]:
        with pdfplumber.open(self.document_file_path) as pdf:
            page = pdf.pages[page_idx]
            text = page.extract_text()

        return {
            "text": text,
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
        }
