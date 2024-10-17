## Load text from PDF files (using PyPDF2)

??? note "Note"
    **Underlying Library:** `pypdf2`

    A pure-python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files

    You can interact with the underlying library and fine-tune the outputs via `**kwargs`.

    Use it in our library with:
    ```python
    from medrag_multi_modal.document_loader.text_loader import PyPDF2TextLoader
    ```

    For details and available `**kwargs`, please refer to the sources below.

    **Sources:**

    - [Docs](https://pypdf2.readthedocs.io/en/3.x/)
    - [GitHub](https://github.com/py-pdf/pypdf)
    - [PyPI](https://pypi.org/project/PyPDF2/)

::: medrag_multi_modal.document_loader.text_loader.pypdf2_text_loader
