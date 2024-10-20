# Load images from PDF files (using PDF2Image)

!!! danger "Warning"
    Unlike other image extraction methods in `document_loader.image_loader`, this loader does not extract embedded images from the PDF.
    Instead, it creates a snapshot image version of each selected page from the PDF.

??? note "Note"
    **Underlying Library:** `pdf2image`

    Extract images from PDF files using `pdf2image`.


    Use it in our library with:
    ```python
    from medrag_multi_modal.document_loader.image_loader import PDF2ImageLoader
    ```

    For details and available `**kwargs`, please refer to the sources below.

    **Sources:**

    - [DataLab](https://www.datalab.to)
    - [GitHub](https://github.com/VikParuchuri/marker)
    - [PyPI](https://pypi.org/project/marker-pdf/)

::: medrag_multi_modal.document_loader.image_loader.pdf2image_img_loader
