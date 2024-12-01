from .fitzpil_img_loader import FitzPILImageLoader

# from .marker_img_loader import MarkerImageLoader
from .pdf2image_img_loader import PDF2ImageLoader
from .pdfplumber_img_loader import PDFPlumberImageLoader
from .pymupdf_img_loader import PyMuPDFImageLoader

__all__ = [
    "PDF2ImageLoader",
    # "MarkerImageLoader",
    "PDFPlumberImageLoader",
    "PyMuPDFImageLoader",
    "FitzPILImageLoader",
]
