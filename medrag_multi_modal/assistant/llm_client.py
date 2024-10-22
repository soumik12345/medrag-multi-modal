import os
from enum import Enum
from typing import Any, Optional, Union

import instructor
import weave
from PIL import Image

from ..utils import base64_encode_image


class ClientType(Enum, str):
    GEMINI = "gemini"
    MISTRAL = "mistral"


class LLMClient(weave.Model):
    model_name: str
    client_type: ClientType

    def __init__(self, model_name: str, client_type: ClientType):
        super().__init__(model_name=model_name, client_type=client_type)

    @weave.op()
    def execute_gemini_sdk(
        self,
        user_prompt: Union[str, list[str]],
        system_prompt: Optional[Union[str, list[str]]] = None,
        schema: Optional[Any] = None,
    ) -> Union[str, Any]:
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
        generation_config = (
            None
            if schema is None
            else genai.GenerationConfig(
                response_mime_type="application/json", response_schema=list[schema]
            )
        )
        response = model.generate_content(
            user_prompt, generation_config=generation_config
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
        messages = [{"type": "text", "text": prompt} for prompt in system_prompt]
        for prompt in user_prompt:
            if isinstance(prompt, Image.Image):
                messages.append(
                    {
                        "type": "image_url",
                        "image_url": base64_encode_image(prompt, "image/png"),
                    }
                )
            else:
                messages.append({"type": "text", "text": prompt})

        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        client = instructor.from_mistral(client)

        response = (
            client.chat.complete(model=self.model_name, messages=messages)
            if schema is None
            else client.messages.create(
                response_model=schema, messages=messages, temperature=0
            )
        )
        return response.choices[0].message.content

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
        else:
            raise ValueError(f"Invalid client type: {self.client_type}")
