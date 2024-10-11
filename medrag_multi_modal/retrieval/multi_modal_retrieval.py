import os

import wandb
import weave
from byaldi import RAGMultiModalModel


class MultiModalRetriever(weave.Model):
    model_name: str
    _docs_retrieval_model: RAGMultiModalModel

    def __init__(self, model_name: str = "vidore/colpali-v1.2"):
        super().__init__(model_name=model_name)
        self._docs_retrieval_model = RAGMultiModalModel.from_pretrained(self.model_name)

    def index(self, data_artifact_name: str, weave_dataset_name: str, index_name: str):
        if wandb.run:
            artifact = wandb.use_artifact(data_artifact_name, type="dataset")
            artifact_dir = artifact.download()
        else:
            api = wandb.Api()
            artifact = api.artifact(data_artifact_name)
            artifact_dir = artifact.download()
        self._docs_retrieval_model.index(
            input_path=artifact_dir,
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
