from pydantic import BaseModel

from medrag_multi_modal.assistant.llm_client import ClientType, LLMClient


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


def test_llm_client():
    llm_client = LLMClient(model_name="gpt-4o-mini", client_type=ClientType.OPENAI)
    event = llm_client.predict(
        system_prompt="Extract the event information",
        user_prompt="Alice and Bob are going to a science fair on Friday.",
        schema=CalendarEvent,
    )
    assert event.name.lower() == "science fair"
    assert event.date.lower() == "friday"
    assert [item.lower() for item in event.participants] == ["alice", "bob"]
