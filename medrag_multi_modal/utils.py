import torch
import wandb


def get_wandb_artifact(
    artifact_name: str, artifact_type: str, get_metadata: bool = False
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
        return "cuda"
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    return "cpu"
