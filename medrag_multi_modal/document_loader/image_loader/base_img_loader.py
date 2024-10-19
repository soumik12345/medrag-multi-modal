import asyncio
import os
from abc import abstractmethod
from typing import Dict, List, Optional

import rich

import wandb
from medrag_multi_modal.document_loader.text_loader.base_text_loader import (
    BaseTextLoader,
)


class BaseImageLoader(BaseTextLoader):
    def __init__(self, url: str, document_name: str, document_file_path: str):
        super().__init__(url, document_name, document_file_path)

    @abstractmethod
    async def extract_page_data(
        self, page_idx: int, image_save_dir: str, **kwargs
    ) -> Dict[str, str]:
        pass

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        wandb_artifact_name: Optional[str] = None,
        image_save_dir: str = "./images",
        cleanup: bool = True,
        **kwargs,
    ) -> List[Dict[str, str]]:
        os.makedirs(image_save_dir, exist_ok=True)
        start_page, end_page = self.get_page_indices(start_page, end_page)
        pages = []
        processed_pages_counter: int = 1
        total_pages = end_page - start_page

        async def process_page(page_idx):
            nonlocal processed_pages_counter
            page_data = await self.extract_page_data(page_idx, image_save_dir, **kwargs)
            pages.append(page_data)
            rich.print(
                f"Processed page idx: {page_idx}, progress: {processed_pages_counter}/{total_pages}"
            )
            processed_pages_counter += 1

        tasks = [process_page(page_idx) for page_idx in range(start_page, end_page)]
        for task in asyncio.as_completed(tasks):
            await task

        if wandb_artifact_name:
            artifact = wandb.Artifact(name=wandb_artifact_name, type="dataset")
            artifact.add_dir(local_path=image_save_dir)
            artifact.save()
            rich.print("Artifact saved and uploaded to wandb!")

        if cleanup:
            for file in os.listdir(image_save_dir):
                file_path = os.path.join(image_save_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        return pages
