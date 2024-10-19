import os
from typing import Any, Dict

from pdf2image.pdf2image import convert_from_path

from .base_img_loader import BaseImageLoader


class PDF2ImageLoader(BaseImageLoader):

    def __init__(self, url: str, document_name: str, document_file_path: str):
        super().__init__(url, document_name, document_file_path)

    async def extract_page_data(
        self, page_idx: int, image_save_dir: str, **kwargs
    ) -> Dict[str, Any]:
        image = convert_from_path(
            self.document_file_path,
            first_page=page_idx + 1,
            last_page=page_idx + 1,
            **kwargs,
        )[0]

        image_file_name = f"page{page_idx}.png"
        image_file_path = os.path.join(image_save_dir, image_file_name)
        image.save(image_file_path)

        return {
            "page_idx": page_idx,
            "document_name": self.document_name,
            "file_path": self.document_file_path,
            "file_url": self.url,
            "image_file_path": image_file_path,
        }
