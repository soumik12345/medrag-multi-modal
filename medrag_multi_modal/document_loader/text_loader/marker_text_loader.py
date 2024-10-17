from typing import Dict

from marker.convert import convert_single_pdf
from marker.models import load_all_models

from .base_text_loader import BaseTextLoader


class MarkerTextLoader(BaseTextLoader):
    async def _process_page(self, page_idx: int) -> Dict[str, str]:
        model_lst = load_all_models()

        text, _, out_meta = convert_single_pdf(
            self.document_file_path,
            model_lst,
            max_pages=1,
            batch_multiplier=1,
            start_page=page_idx,
            ocr_all_pages=True,
        )

        return {
            "text": text,
            "meta": out_meta,
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
        }
