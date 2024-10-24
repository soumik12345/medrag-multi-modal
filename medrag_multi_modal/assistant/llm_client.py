import os
from enum import Enum
from typing import Any, Optional, Union

import instructor
import weave
from PIL import Image

from ..utils import base64_encode_image


class ClientType(str, Enum):
    GEMINI = "gemini"
    MISTRAL = "mistral"
    OPENAI = "openai"


GOOGLE_MODELS = [
    "gemini-1.0-pro-latest",
    "gemini-1.0-pro",
    "gemini-pro",
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-vision-latest",
    "gemini-pro-vision",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro",
    "gemini-1.5-pro-exp-0801",
    "gemini-1.5-pro-exp-0827",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-001-tuning",
    "gemini-1.5-flash",
    "gemini-1.5-flash-exp-0827",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b-001",
    "gemini-1.5-flash-8b-latest",
    "gemini-1.5-flash-8b-exp-0827",
    "gemini-1.5-flash-8b-exp-0924",
]

MISTRAL_MODELS = [
    "ministral-3b-latest",
    "ministral-8b-latest",
    "mistral-large-latest",
    "mistral-small-latest",
    "codestral-latest",
    "pixtral-12b-2409",
    "open-mistral-nemo",
    "open-codestral-mamba",
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "open-mixtral-8x22b",
]

OPENAI_MODELS = ["gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-4o-mini-2024-07-18"]


class LLMClient(weave.Model):
    model_name: str
    client_type: Optional[ClientType]

    def __init__(self, model_name: str, client_type: Optional[ClientType] = None):
        if client_type is None:
            if model_name in GOOGLE_MODELS:
                client_type = ClientType.GEMINI
            elif model_name in MISTRAL_MODELS:
                client_type = ClientType.MISTRAL
            elif model_name in OPENAI_MODELS:
                client_type = ClientType.OPENAI
            else:
                raise ValueError(f"Invalid model name: {model_name}")
        super().__init__(model_name=model_name, client_type=client_type)

    @weave.op()
    def execute_gemini_sdk(
        self,
        user_prompt: Union[str, list[str]],
        system_prompt: Optional[Union[str, list[str]]] = None,
        schema: Optional[Any] = None,
    ) -> Union[str, Any]:
        import google.generativeai as genai

        system_prompt = (
            [system_prompt] if isinstance(system_prompt, str) else system_prompt
        )
        user_prompt = [user_prompt] if isinstance(user_prompt, str) else user_prompt

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(self.model_name)
        generation_config = (
            None
            if schema is None
            else genai.GenerationConfig(
                response_mime_type="application/json", response_schema=list[schema]
            )
        )
        response = model.generate_content(
            system_prompt + user_prompt, generation_config=generation_config
        )
        return response.text if schema is None else response

    @weave.op()
    def execute_mistral_sdk(
        self,
        user_prompt: Union[str, list[str]],
        system_prompt: Optional[Union[str, list[str]]] = None,
        schema: Optional[Any] = None,
    ) -> Union[str, Any]:
        from mistralai import Mistral

        system_prompt = (
            [system_prompt] if isinstance(system_prompt, str) else system_prompt
        )
        user_prompt = [user_prompt] if isinstance(user_prompt, str) else user_prompt
        system_messages = [{"type": "text", "text": prompt} for prompt in system_prompt]
        user_messages = []
        for prompt in user_prompt:
            if isinstance(prompt, Image.Image):
                user_messages.append(
                    {
                        "type": "image_url",
                        "image_url": base64_encode_image(prompt, "image/png"),
                    }
                )
            else:
                user_messages.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": system_messages},
            {"role": "user", "content": user_messages},
        ]

        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        client = instructor.from_mistral(client) if schema is not None else client

        response = (
            client.chat.complete(model=self.model_name, messages=messages)
            if schema is None
            else client.messages.create(
                response_model=schema, messages=messages, temperature=0
            )
        )
        return response.choices[0].message.content

    @weave.op()
    def execute_openai_sdk(
        self,
        user_prompt: Union[str, list[str]],
        system_prompt: Optional[Union[str, list[str]]] = None,
        schema: Optional[Any] = None,
    ) -> Union[str, Any]:
        from openai import OpenAI

        system_prompt = (
            [system_prompt] if isinstance(system_prompt, str) else system_prompt
        )
        user_prompt = [user_prompt] if isinstance(user_prompt, str) else user_prompt

        system_messages = [
            {"role": "system", "content": prompt} for prompt in system_prompt
        ]
        user_messages = []
        for prompt in user_prompt:
            if isinstance(prompt, Image.Image):
                user_messages.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_encode_image(prompt, "image/png"),
                        },
                    },
                )
            else:
                user_messages.append({"type": "text", "text": prompt})
        messages = system_messages + [{"role": "user", "content": user_messages}]

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        if schema is None:
            completion = client.chat.completions.create(
                model=self.model_name, messages=messages
            )
            return completion.choices[0].message.content

        completion = weave.op()(client.beta.chat.completions.parse)(
            model=self.model_name, messages=messages, response_format=schema
        )
        return completion.choices[0].message.parsed

    @weave.op()
    def predict(
        self,
        user_prompt: Union[str, list[str]],
        system_prompt: Optional[Union[str, list[str]]] = None,
        schema: Optional[Any] = None,
    ) -> Union[str, Any]:
        if self.client_type == ClientType.GEMINI:
            return self.execute_gemini_sdk(user_prompt, system_prompt, schema)
        elif self.client_type == ClientType.MISTRAL:
            return self.execute_mistral_sdk(user_prompt, system_prompt, schema)
        elif self.client_type == ClientType.OPENAI:
            return self.execute_openai_sdk(user_prompt, system_prompt, schema)
        else:
            raise ValueError(f"Invalid client type: {self.client_type}")
