import asyncio
import os
from abc import abstractmethod
from glob import glob
from typing import Dict, List, Optional

import huggingface_hub
import jsonlines
import rich
from datasets import (
    Dataset,
    Features,
    Image,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)

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
        """
        Abstract method to process a single page of the PDF and extract the image data.

        Overwrite this method in the subclass to provide the actual implementation and
        processing logic for each page of the PDF using various PDF processing libraries.

        Args:
            page_idx (int): The index of the page to process.
            image_save_dir (str): The directory to save the extracted images.
            **kwargs: Additional keyword arguments that may be used by underlying libraries.

        Returns:
            Dict[str, str]: A dictionary containing the processed page data.
        """
        pass

    def save_as_dataset(
        self,
        start_page: int,
        end_page: int,
        image_save_dir: str,
        dataset_repo_id: Optional[str] = None,
        overwrite_dataset: bool = False,
    ):
        features = Features(
            {
                "page_image": Image(decode=True),
                "page_figure_images": Sequence(Image(decode=True)),
                "document_name": Value(dtype="string"),
                "page_idx": Value(dtype="int32"),
            }
        )

        all_examples = []
        for page_idx in range(start_page, end_page):
            page_image_file_paths = glob(
                os.path.join(image_save_dir, f"page{page_idx}*.png")
            )
            if len(page_image_file_paths) > 0:
                page_image_path = page_image_file_paths[0]
                figure_image_paths = [
                    image_file_path
                    for image_file_path in glob(
                        os.path.join(image_save_dir, f"page{page_idx}*_fig*.png")
                    )
                ]

                example = {
                    "page_image": page_image_path,
                    "page_figure_images": figure_image_paths,
                    "document_name": self.document_name,
                    "page_idx": page_idx,
                }
                all_examples.append(example)

        dataset = Dataset.from_list(all_examples, features=features)

        if dataset_repo_id:
            if huggingface_hub.repo_exists(dataset_repo_id, repo_type="dataset"):
                if not overwrite_dataset:
                    dataset = concatenate_datasets(
                        [dataset, load_dataset(dataset_repo_id)["corpus"]]
                    )

            dataset.push_to_hub(dataset_repo_id, split="corpus")

        return dataset

    def cleanup_image_dir(self, image_save_dir: str = "./images"):
        for file in os.listdir(image_save_dir):
            file_path = os.path.join(image_save_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    async def load_data(
        self,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        dataset_repo_id: Optional[str] = None,
        overwrite_dataset: bool = False,
        image_save_dir: str = "./images",
        exclude_file_extensions: list[str] = [],
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Asynchronously loads images from a PDF file specified by a URL or local file path.
        The overrided processing abstract method then processes the images,
        and optionally publishes it to a WandB artifact.

        This function downloads a PDF from a given URL if it does not already exist locally,
        reads the specified range of pages, scans each page's content to extract images, and
        returns a list of Page objects containing the images and metadata.

        It uses `PyPDF2` to calculate the number of pages in the PDF and the
        overriden `extract_page_data` method provides the actual implementation to process
        each page, extract the image content from the PDF, and convert it to png format.
        It processes pages concurrently using `asyncio` for efficiency.

        If a wandb_artifact_name is provided, the processed pages are published to a WandB artifact.

        Args:
            start_page (Optional[int]): The starting page index (0-based) to process.
            end_page (Optional[int]): The ending page index (0-based) to process.
            dataset_repo_id (Optional[str]): The repository ID of the HuggingFace dataset to publish the pages to, if provided.
            overwrite_dataset (bool): Whether to overwrite the existing dataset if it exists. Defaults to False.
            image_save_dir (str): The directory to save the extracted images.
            exclude_file_extensions (list[str]): A list of file extensions to exclude from the image_save_dir.
            **kwargs: Additional keyword arguments that will be passed to extract_page_data method and the underlying library.

        Returns:
            Dataset: A HuggingFace dataset containing the processed pages.

        Raises:
            ValueError: If the specified start_page or end_page is out of bounds of the document's page count.
        """
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

        with jsonlines.open(
            os.path.join(image_save_dir, "metadata.jsonl"), mode="w"
        ) as writer:
            writer.write(pages)

        for file in os.listdir(image_save_dir):
            if file.endswith(tuple(exclude_file_extensions)):
                os.remove(os.path.join(image_save_dir, file))

        dataset = self.save_as_dataset(
            start_page, end_page, image_save_dir, dataset_repo_id, overwrite_dataset
        )

        return dataset
