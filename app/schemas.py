from datetime import datetime
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    was_blocked: bool = False


class ChatHistoryItem(BaseModel):
    role: str
    content: str
    was_blocked: bool
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatHistoryItem]
