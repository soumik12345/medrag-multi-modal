from PIL import Image
from pydantic import BaseModel

from medrag_multi_modal.assistant.llm_client import ClientType, LLMClient


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


class ImageDescription(BaseModel):
    description: str


def test_openai_llm_client():
    llm_client = LLMClient(model_name="gpt-4o-mini", client_type=ClientType.OPENAI)
    event = llm_client.predict(
        system_prompt="Extract the event information",
        user_prompt="Alice and Bob are going to a science fair on Friday.",
        schema=CalendarEvent,
    )
    assert event.name.lower() == "science fair"
    assert event.date.lower() == "friday"
    assert [item.lower() for item in event.participants] == ["alice", "bob"]


def test_openai_image_description():
    llm_client = LLMClient(model_name="gpt-4o-mini", client_type=ClientType.OPENAI)
    description = llm_client.predict(
        system_prompt="Describe the image",
        user_prompt=[Image.open("./assets/test_image.png")],
        schema=ImageDescription,
    )
    assert "astronaut" in description.description.lower()


def test_mistral_llm_client():
    llm_client = LLMClient(model_name="ministral-3b-latest", client_type=ClientType.MISTRAL)
    event = llm_client.predict(
        system_prompt="Extract the event information",
        user_prompt="Alice and Bob are going to a science fair on Friday.",
        schema=CalendarEvent,
    )
    assert event.name.lower() == "science fair"
    assert event.date.lower() == "friday"
    assert [item.lower() for item in event.participants] == ["alice", "bob"]
