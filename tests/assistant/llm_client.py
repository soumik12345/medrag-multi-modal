from pydantic import BaseModel
from medrag_multi_modal.assistant.llm_client import LLMClient, ClientType


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


def test_llm_client():
    llm_client = LLMClient(client_type=ClientType.OPENAI)
    event = llm_client.predict(
        system_prompt="Extract the event information",
        user_prompt="Alice and Bob are going to a science fair on Friday.",
        schema=CalendarEvent,
    )
    assert event.name == "science fair"
    assert event.date == "Friday"
    assert event.participants == ["Alice", "Bob"]
