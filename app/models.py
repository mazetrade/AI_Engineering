from pydantic import BaseModel
from typing import Optional

class UserMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class BotResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    action: str   # "continue" | "escalate" | "out_of_scope"       # What the bot understood the user wants