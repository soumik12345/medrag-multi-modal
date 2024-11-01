import asyncio

from medrag_multi_modal.document_loader.image_loader import FitzPILImageLoader

URL = "https://archive.org/download/GraysAnatomy41E2015PDF/Grays%20Anatomy-41%20E%20%282015%29%20%5BPDF%5D.pdf"


def test_fitzpil_img_loader():
    loader = FitzPILImageLoader(
        url=URL,
        document_name="Gray's Anatomy",
        document_file_path="grays_anatomy.pdf",
    )
    dataset = asyncio.run(
        loader.load_data(
            start_page=32,
            end_page=37,
            wandb_artifact_name="grays-anatomy-images-fitzpil",
            cleanup=False,
        )
    )
