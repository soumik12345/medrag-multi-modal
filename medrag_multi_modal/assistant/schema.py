from pydantic import BaseModel


class FigureAnnotation(BaseModel):
    figure_id: str
    figure_description: str


class FigureAnnotations(BaseModel):
    annotations: list[FigureAnnotation]
