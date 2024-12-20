
from pydantic import BaseModel
from typing import List


class Document(BaseModel):
    file_name: str | None
    index_id: str | None
    
class Account(BaseModel):
    user_name: str
    documents: List[Document]  | None

