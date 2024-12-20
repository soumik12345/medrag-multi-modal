import os
from glob import glob
from typing import Optional, Union

import cv2
import weave
from PIL import Image

from medrag_multi_modal.assistant.llm_client import LLMClient
from medrag_multi_modal.assistant.schema import FigureAnnotations
from medrag_multi_modal.utils import get_wandb_artifact, read_jsonl_file


class FigureAnnotatorFromPageImage(weave.Model):
    """
    `FigureAnnotatorFromPageImage` is a class that leverages two LLM clients to annotate
    figures from a page image of a scientific textbook.

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
            image_artifact_address="ml-colabs/medrag-multi-modal/grays-anatomy-images-marker:v6",
        )
        annotations = figure_annotator.predict(page_idx=34)
        ```

    Args:
        figure_extraction_llm_client (LLMClient): An LLM client used to extract figure annotations
            from the page image.
        structured_output_llm_client (LLMClient): An LLM client used to convert the extracted
            annotations into a structured format.
        image_artifact_address (Optional[str]): The address of the image artifact containing the
            page images.
    """

    figure_extraction_llm_client: LLMClient
    structured_output_llm_client: LLMClient
    _artifact_dir: str

    def __init__(
        self,
        figure_extraction_llm_client: LLMClient,
        structured_output_llm_client: LLMClient,
        image_artifact_address: Optional[str] = None,
    ):
        super().__init__(
            figure_extraction_llm_client=figure_extraction_llm_client,
            structured_output_llm_client=structured_output_llm_client,
        )
        self._artifact_dir = get_wandb_artifact(image_artifact_address, "dataset")

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
    def predict(self, page_idx: int) -> dict[int, list[FigureAnnotations]]:
        """
        Predicts figure annotations for a specific page in a document.

        This function retrieves the artifact directory from the given image artifact address,
        reads the metadata from the 'metadata.jsonl' file, and iterates through the metadata
        to find the specified page index. If the page index matches, it reads the page image
        and associated figure images, and then uses the `annotate_figures` method to extract
        figure annotations from the page image. The extracted annotations are then structured
        using the `extract_structured_output` method and returned as a dictionary.

        Args:
            page_idx (int): The index of the page to annotate.
            image_artifact_address (str): The address of the image artifact containing the
                page images.

        Returns:
            dict: A dictionary containing the page index as the key and the extracted figure
                annotations as the value.
        """

        metadata = read_jsonl_file(os.path.join(self._artifact_dir, "metadata.jsonl"))
        annotations = {}
        for item in metadata:
            if item["page_idx"] == page_idx:
                page_image_file = os.path.join(
                    self._artifact_dir, f"page{item['page_idx']}.png"
                )
                figure_image_files = glob(
                    os.path.join(self._artifact_dir, f"page{item['page_idx']}_fig*.png")
                )
                if len(figure_image_files) > 0:
                    page_image = cv2.imread(page_image_file)
                    page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
                    page_image = Image.fromarray(page_image)
                    figure_extracted_annotations = self.annotate_figures(
                        page_image=page_image
                    )
                    figure_extracted_annotations = self.extract_structured_output(
                        figure_extracted_annotations["annotations"]
                    ).model_dump()
                    annotations[item["page_idx"]] = figure_extracted_annotations[
                        "annotations"
                    ]
                break
        return annotations
