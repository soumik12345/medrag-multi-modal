import asyncio
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import PyPDF2
import rich
import weave
from firerequests import FireRequests


class BaseTextLoader(ABC):
    def __init__(self, url: str, document_name: str, document_file_path: str):
        self.url = url
        self.document_name = document_name
        self.document_file_path = document_file_path
        if not os.path.exists(self.document_file_path):
            FireRequests().download(url, filename=self.document_file_path)
        with open(self.document_file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            self.page_count = len(pdf_reader.pages)

    def get_page_indices(
        self, start_page: Optional[int] = None, end_page: Optional[int] = None
    ) -> tuple[int, int]:
        if start_page:
            if start_page > self.page_count:
                raise ValueError(
                    f"Start page {start_page} is greater than the total page count {self.page_count}"
                )
        else:
            start_page = 0
        if end_page:
            if end_page > self.page_count:
                raise ValueError(
                    f"End page {end_page} is greater than the total page count {self.page_count}"
                )
        else:
            end_page = self.page_count - 1
        return start_page, end_page

    @abstractmethod
    async def _process_page(self, page_idx: int) -> Dict[str, str]:
        pass

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        weave_dataset_name: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        start_page, end_page = self.get_page_indices(start_page, end_page)
        pages = []
        processed_pages_counter: int = 1
        total_pages = end_page - start_page

        async def process_page(page_idx):
            nonlocal processed_pages_counter
            page_data = await self._process_page(page_idx)
            pages.append(page_data)
            rich.print(
                f"Processed page idx: {page_idx}, progress: {processed_pages_counter}/{total_pages}"
            )
            processed_pages_counter += 1

        tasks = [process_page(page_idx) for page_idx in range(start_page, end_page)]
        for task in asyncio.as_completed(tasks):
            await task

        if weave_dataset_name:
            weave.publish(weave.Dataset(name=weave_dataset_name, rows=pages))
        return pages
