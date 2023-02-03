from typing import List

from pydantic import BaseModel


class ProcessedDatum(BaseModel):
    __root__: List["ProcessedDataModel"]


class ProcessedDataModel(BaseModel):
    title: str
    chapter: str
    creator: str
    text: str
    questions: List[str] = []


ProcessedDatum.update_forward_refs()