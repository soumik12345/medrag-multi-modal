## Load text from PDF files (using PDFPlumber)

??? note "Note"
    **Underlying Library:** `pdfplumber`

    Plumb a PDF for detailed information about each char, rectangle, line, et cetera — and easily extract text and tables.

    You can interact with the underlying library and fine-tune the outputs via `**kwargs`.

    Use it in our library with:
    ```python
    from medrag_multi_modal.document_loader.text_loader import PDFPlumberTextLoader
    ```

    For details and available `**kwargs`, please refer to the sources below.

    **Sources:**

    - [GitHub](https://github.com/jsvine/pdfplumber)
    - [PyPI](https://pypi.org/project/pdfplumber/)

::: medrag_multi_modal.document_loader.text_loader.pdfplumber_text_loader
