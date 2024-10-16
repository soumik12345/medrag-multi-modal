from typing import Dict

import pymupdf4llm

from .base_text_loader import BaseTextLoader


class PyMuPDF4LLMTextLoader(BaseTextLoader):
    async def _process_page(self, page_idx: int) -> Dict[str, str]:
        text = pymupdf4llm.to_markdown(
            doc=self.document_file_path, pages=[page_idx], show_progress=False
        )
        return {
            "text": text,
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
        }
