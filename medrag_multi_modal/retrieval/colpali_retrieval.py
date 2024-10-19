import os
from typing import Any, Optional

import weave
from byaldi import RAGMultiModalModel
from PIL import Image

import wandb

from ..utils import get_wandb_artifact


class CalPaliRetriever(weave.Model):
    """
    CalPaliRetriever is a class that facilitates the retrieval of page images using ColPali.

    This class leverages the `byaldi.RAGMultiModalModel` to perform document retrieval tasks.
    It can be initialized with a pre-trained model or from a specified W&B artifact. The class
    also provides methods to index new data and to predict/retrieve documents based on a query.

    !!! example "Indexing Data"
        First you need to install `Byaldi` library by Answer.ai.
        
        ```bash
        uv pip install Byaldi>=0.0.5
        ```
        
        Next, you can index the data by running the following code:
        
        ```python
        import wandb
        from medrag_multi_modal.retrieval import CalPaliRetriever

        wandb.init(project="medrag-multi-modal", entity="ml-colabs", job_type="index")
        retriever = CalPaliRetriever()
        retriever.index(
            data_artifact_name="ml-colabs/medrag-multi-modal/grays-anatomy-images:v1",
            weave_dataset_name="grays-anatomy-images:v0",
            index_name="grays-anatomy",
        )
        ```

    !!! example "Retrieving Documents"
        First you need to install `Byaldi` library by Answer.ai.
        
        ```bash
        uv pip install Byaldi>=0.0.5
        ```
        
        Next, you can retrieve the documents by running the following code:
        
        ```python
        import weave

        import wandb
        from medrag_multi_modal.retrieval import CalPaliRetriever

        weave.init(project_name="ml-colabs/medrag-multi-modal")
        retriever = CalPaliRetriever.from_artifact(
            index_artifact_name="ml-colabs/medrag-multi-modal/grays-anatomy:v0",
            metadata_dataset_name="grays-anatomy-images:v0",
            data_artifact_name="ml-colabs/medrag-multi-modal/grays-anatomy-images:v1",
        )
        retriever.predict(
            query="which neurotransmitters convey information between Merkel cells and sensory afferents?",
            top_k=3,
        )
        ```

    Attributes:
        model_name (str): The name of the model to be used for retrieval.
    """

    model_name: str
    _docs_retrieval_model: Optional[RAGMultiModalModel] = None
    _metadata: Optional[dict] = None
    _data_artifact_dir: Optional[str] = None

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        docs_retrieval_model: Optional[RAGMultiModalModel] = None,
        data_artifact_dir: Optional[str] = None,
        metadata_dataset_name: Optional[str] = None,
    ):
        super().__init__(model_name=model_name)
        self._docs_retrieval_model = (
            docs_retrieval_model or RAGMultiModalModel.from_pretrained(self.model_name)
        )
        self._data_artifact_dir = data_artifact_dir
        self._metadata = (
            [dict(row) for row in weave.ref(metadata_dataset_name).get().rows]
            if metadata_dataset_name
            else None
        )

    @classmethod
    def from_artifact(
        cls,
        index_artifact_name: str,
        metadata_dataset_name: str,
        data_artifact_name: str,
    ):
        index_artifact_dir = get_wandb_artifact(index_artifact_name, "colpali-index")
        data_artifact_dir = get_wandb_artifact(data_artifact_name, "dataset")
        docs_retrieval_model = RAGMultiModalModel.from_index(
            index_path=os.path.join(index_artifact_dir, "index")
        )
        return cls(
            docs_retrieval_model=docs_retrieval_model,
            metadata_dataset_name=metadata_dataset_name,
            data_artifact_dir=data_artifact_dir,
        )

    def index(self, data_artifact_name: str, weave_dataset_name: str, index_name: str):
        data_artifact_dir = get_wandb_artifact(data_artifact_name, "dataset")
        self._docs_retrieval_model.index(
            input_path=data_artifact_dir,
            index_name=index_name,
            store_collection_with_index=False,
            overwrite=True,
        )
        if wandb.run:
            artifact = wandb.Artifact(
                name=index_name,
                type="colpali-index",
                metadata={"weave_dataset_name": weave_dataset_name},
            )
            artifact.add_dir(
                local_path=os.path.join(".byaldi", index_name), name="index"
            )
            artifact.save()

    @weave.op()
    def predict(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Predicts and retrieves the top-k most relevant documents/images for a given query
        using ColPali.

        This function uses the document retrieval model to search for the most relevant
        documents based on the provided query. It returns a list of dictionaries, each
        containing the document image, document ID, and the relevance score.

        Args:
            query (str): The search query string.
            top_k (int, optional): The number of top results to retrieve. Defaults to 10.

        Returns:
            list[dict[str, Any]]: A list of dictionaries where each dictionary contains:
                - "doc_image" (PIL.Image.Image): The image of the document.
                - "doc_id" (str): The ID of the document.
                - "score" (float): The relevance score of the document.
        """
        results = self._docs_retrieval_model.search(query=query, k=top_k)
        retrieved_results = []
        for result in results:
            retrieved_results.append(
                {
                    "doc_image": Image.open(
                        os.path.join(self._data_artifact_dir, f"{result['doc_id']}.png")
                    ),
                    "doc_id": result["doc_id"],
                    "score": result["score"],
                }
            )
        return retrieved_results
