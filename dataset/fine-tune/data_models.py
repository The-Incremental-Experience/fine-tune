from typing import List

from pydantic import BaseModel


class ProcessedDataModel(BaseModel):
    title: str
    text: str
    questions: List[str] = []
