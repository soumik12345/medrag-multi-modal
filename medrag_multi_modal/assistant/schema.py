from typing import Union

from pydantic import BaseModel


class FigureAnnotation(BaseModel):
    figure_id: str
    figure_description: str


class FigureAnnotations(BaseModel):
    annotations: list[FigureAnnotation]


class MedQAMCQResponse(BaseModel):
    answer: str
    explanation: str


class MedQACitation(BaseModel):
    page_number: int
    document_name: str


class MedQAResponse(BaseModel):
    response: Union[str, MedQAMCQResponse]
    citations: list[str]
