import base64
import io

import jsonlines
import torch
from huggingface_hub import HfApi
from PIL import Image

import wandb


def get_wandb_artifact(
    artifact_name: str,
    artifact_type: str,
    get_metadata: bool = False,
) -> str:
    if wandb.run:
        artifact = wandb.use_artifact(artifact_name, type=artifact_type)
        artifact_dir = artifact.download()
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download()
    if get_metadata:
        return artifact_dir, artifact.metadata
    return artifact_dir


def get_torch_backend():
    if torch.cuda.is_available():
        if torch.backends.cuda.is_built():
            return "cuda"
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    return "cpu"


def base64_encode_image(image: Image.Image, mimetype: str) -> str:
    image.load()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    encoded_string = f"data:{mimetype};base64,{encoded_string}"
    return str(encoded_string)


def read_jsonl_file(file_path: str) -> list[dict[str, any]]:
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            return obj


def save_to_huggingface(
    repo_id: str, local_dir: str, commit_message: str, private: bool = False
):
    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        token=api.token,
        private=private,
        repo_type="model",
        exist_ok=True,
    )
    repo_id = repo_url.repo_id
    api.upload_folder(
        repo_id=repo_id,
        commit_message=commit_message,
        token=api.token,
        folder_path=local_dir,
        repo_type=repo_url.repo_type,
    )


def fetch_from_huggingface(repo_id: str, local_dir: str) -> str:
    api = HfApi()
    repo_url = api.repo_info(repo_id)
    if repo_url is None:
        raise ValueError(f"Model {repo_id} not found on the Hugging Face Hub.")

    snapshot = api.snapshot_download(repo_id, revision=None, local_dir=local_dir)
    if snapshot is None:
        raise ValueError(f"Model {repo_id} not found on the Hugging Face Hub.")
    return snapshot
