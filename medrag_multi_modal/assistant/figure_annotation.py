import os
from typing import Union

import cv2
import weave
from PIL import Image
from rich.progress import track

from ..utils import get_wandb_artifact, read_jsonl_file
from .llm_client import LLMClient


class FigureAnnotatorFromPageImage(weave.Model):
    llm_client: LLMClient

    @weave.op()
    def annotate_figures(
        self, page_image: Image.Image
    ) -> dict[str, Union[Image.Image, str]]:
        annotation = self.llm_client.predict(
            system_prompt="""
You are an expert in the domain of scientific textbooks, especially medical texts.
You are presented with a page from a scientific textbook.
You are to first identify the number of figures in the image.
Then you are to identify the figure IDs associated with each figure in the image.
Then, you are to extract the exact figure descriptions from the image.
You need to output the figure IDs and descriptions in a structured manner as a JSON object.

Here are some clues you need to follow:
1. Figure IDs are unique identifiers for each figure in the image.
2. Sometimes figure IDs can also be found as captions to the immediate left, right, top, or bottom of the figure.
3. Figure IDs are in the form "Fig X.Y" where X and Y are integers. For example, 1.1, 1.2, 1.3, etc.
4. Figure descriptions are contained as captions under the figures in the image, just after the figure ID.
5. The text in the image is written in English and is present in a two-column format.
6. There is a clear distinction between the figure caption and the regular text in the image in the form of extra white space.
7. There might be multiple figures present in the image.
8. The figures may or may not have a distinct border against a white background.
9. There might be multiple figures present in the image. You are to carefully identify all the figures in the image.
""",
            user_prompt=[page_image],
        )
        return {"page_image": page_image, "annotations": annotation}

    @weave.op()
    def predict(self, image_artifact_address: str):
        artifact_dir = get_wandb_artifact(image_artifact_address, "dataset")
        metadata = read_jsonl_file(os.path.join(artifact_dir, "metadata.jsonl"))
        annotations = []
        for item in track(metadata, description="Annotating images:"):
            page_image = cv2.imread(
                os.path.join(artifact_dir, f"page{item['page_idx']}.png")
            )
            page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
            page_image = Image.fromarray(page_image)
            annotations.append(self.annotate_figures(page_image=page_image))
        return annotations
