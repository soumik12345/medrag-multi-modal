import os
from glob import glob
from typing import Union

import cv2
import weave
from PIL import Image
from pydantic import BaseModel
from rich.progress import track

from ..utils import get_wandb_artifact, read_jsonl_file
from .llm_client import LLMClient


class FigureAnnotation(BaseModel):
    figure_id: str
    figure_description: str


class FigureAnnotations(BaseModel):
    annotations: list[FigureAnnotation]


class FigureAnnotatorFromPageImage(weave.Model):
    """
    `FigureAnnotatorFromPageImage` is a class that leverages two LLM clients to annotate figures from a page image of a scientific textbook.

    !!! example "Example Usage"
        ```python
        import weave
        from dotenv import load_dotenv

        from medrag_multi_modal.assistant import (
            FigureAnnotatorFromPageImage, LLMClient
        )

        load_dotenv()
        weave.init(project_name="ml-colabs/medrag-multi-modal")
        figure_annotator = FigureAnnotatorFromPageImage(
            figure_extraction_llm_client=LLMClient(model_name="pixtral-12b-2409"),
            structured_output_llm_client=LLMClient(model_name="gpt-4o"),
        )
        annotations = figure_annotator.predict(
            image_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-images-marker:v6"
        )
        ```

    Attributes:
        figure_extraction_llm_client (LLMClient): An LLM client used to extract figure annotations from the page image.
        structured_output_llm_client (LLMClient): An LLM client used to convert the extracted annotations into a structured format.
    """

    figure_extraction_llm_client: LLMClient
    structured_output_llm_client: LLMClient

    @weave.op()
    def annotate_figures(
        self, page_image: Image.Image
    ) -> dict[str, Union[Image.Image, str]]:
        annotation = self.figure_extraction_llm_client.predict(
            system_prompt="""
You are an expert in the domain of scientific textbooks, especially medical texts.
You are presented with a page from a scientific textbook from the domain of biology, specifically anatomy.
You are to first identify all the figures in the page image, which could be images or biological diagrams, charts, graphs, etc.
Then you are to identify the figure IDs associated with each figure in the page image.
Then, you are to extract only the exact figure descriptions from the page image.
You need to output the figure IDs and figure descriptions only, in a structured manner as a JSON object.

Here are some clues you need to follow:
1. Figure IDs are unique identifiers for each figure in the page image.
2. Sometimes figure IDs can also be found as captions to the immediate left, right, top, or bottom of the figure.
3. Figure IDs are in the form "Fig X.Y" where X and Y are integers. For example, 1.1, 1.2, 1.3, etc.
4. Figure descriptions are contained as captions under the figures in the image, just after the figure ID.
5. The text in the page image is written in English and is present in a two-column format.
6. There is a clear distinction between the figure caption and the regular text in the page image in the form of extra white space.
    You are to carefully identify all the figures in the page image.
7. There might be multiple figures or even no figures present in the page image. Sometimes the figures can be present side-by-side
    or one above the other.
8. The figures may or may not have a distinct border against a white background.
10. You are not supposed to alter the figure description in any way present in the page image and you are to extract it as is.
""",
            user_prompt=[page_image],
        )
        return {"page_image": page_image, "annotations": annotation}

    @weave.op
    def extract_structured_output(self, annotations: str) -> FigureAnnotations:
        return self.structured_output_llm_client.predict(
            system_prompt="You are suppossed to extract a list of figure annotations consisting of figure IDs and corresponding figure descriptions.",
            user_prompt=[annotations],
            schema=FigureAnnotations,
        )

    @weave.op()
    def predict(self, image_artifact_address: str):
        """
        Predicts figure annotations for images in a given artifact directory.

        This function retrieves an artifact directory using the provided image artifact address.
        It reads metadata from a JSONL file in the artifact directory and iterates over each item in the metadata.
        For each item, it constructs the file path for the page image and checks for the presence of figure image files.
        If figure image files are found, it reads and converts the page image, then uses the `annotate_figures` method
        to extract figure annotations from the page image. The extracted annotations are then structured using the
        `extract_structured_output` method and appended to the annotations list.

        Args:
            image_artifact_address (str): The address of the image artifact.

        Returns:
            list: A list of dictionaries containing page indices and their corresponding figure annotations.
        """
        artifact_dir = get_wandb_artifact(image_artifact_address, "dataset")
        metadata = read_jsonl_file(os.path.join(artifact_dir, "metadata.jsonl"))
        annotations = []
        for item in track(metadata, description="Annotating images:"):
            page_image_file = os.path.join(artifact_dir, f"page{item['page_idx']}.png")
            figure_image_files = glob(
                os.path.join(artifact_dir, f"page{item['page_idx']}_fig*.png")
            )
            if len(figure_image_files) > 0:
                page_image = cv2.imread(page_image_file)
                page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
                page_image = Image.fromarray(page_image)
                figure_extracted_annotations = self.annotate_figures(
                    page_image=page_image
                )
                annotations.append(
                    {
                        "page_idx": item["page_idx"],
                        "annotations": self.extract_structured_output(
                            figure_extracted_annotations["annotations"]
                        ).model_dump(),
                    }
                )
        return annotations
