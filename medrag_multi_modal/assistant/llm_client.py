import os
from enum import Enum
from typing import Any, Optional, Union

import google.generativeai as genai
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
    """
    LLMClient is a class that interfaces with different large language model (LLM) providers
    such as Google Gemini, Mistral, and OpenAI. It abstracts the complexity of interacting with
    these different APIs and provides a unified interface for making predictions.

    Args:
        model_name (str): The name of the model to be used for predictions.
        client_type (Optional[ClientType]): The type of client (e.g., GEMINI, MISTRAL, OPENAI).
            If not provided, it is inferred from the model_name.
        publish_system_prompt_to_weave (bool): Whether to publish the system prompt to Weave.
        publish_message_history_to_weave (bool): Whether to publish the message history to Weave.
    """

    model_name: str
    client_type: Optional[ClientType]
    publish_system_prompt_to_weave: bool
    message_history: Union[list[dict], genai.ChatSession] = []
    publish_message_history_to_weave: bool = False

    def __init__(
        self,
        model_name: str,
        client_type: Optional[ClientType] = None,
        publish_system_prompt_to_weave: bool = True,
        publish_message_history_to_weave: bool = False,
    ):
        if client_type is None:
            if model_name in GOOGLE_MODELS:
                client_type = ClientType.GEMINI
            elif model_name in MISTRAL_MODELS:
                client_type = ClientType.MISTRAL
            elif model_name in OPENAI_MODELS:
                client_type = ClientType.OPENAI
            else:
                raise ValueError(f"Invalid model name: {model_name}")
        super().__init__(
            model_name=model_name,
            client_type=client_type,
            publish_system_prompt_to_weave=publish_system_prompt_to_weave,
            publish_message_history_to_weave=publish_message_history_to_weave,
        )

    @weave.op()
    def execute_gemini_sdk(
        self,
        user_prompt: Union[str, list[str]],
        system_prompt: Optional[Union[str, list[str]]] = None,
        schema: Optional[Any] = None,
    ) -> Union[str, Any]:
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        def get_chat_response(
            chat: genai.ChatSession, prompt, generation_config
        ) -> str:
            text_response = []
            responses = chat.send_message(prompt)
            for chunk in responses:
                text_response.append(chunk.text)
            return "".join(text_response)

        system_prompt = (
            [system_prompt] if isinstance(system_prompt, str) else system_prompt
        )
        user_prompt = [user_prompt] if isinstance(user_prompt, str) else user_prompt

        if self.publish_system_prompt_to_weave:
            ref = weave.publish(
                weave.MessagesPrompt(
                    [{"system_prompt": prompt} for prompt in system_prompt]
                ),
                name="medqa_system_prompt_gemini",
            )
            system_prompt_obj = (
                weave.ref(
                    f"weave:///{ref.entity}/{ref.project}/object/{ref.name}:{ref._digest}"
                )
                .get()
                .format()
            )
            system_prompt = [obj["system_prompt"] for obj in system_prompt_obj]

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system_prompt,
            # This is necessary in order to answer questions about anatomy, sexual diseases,
            # medical devices, medicines, etc.
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        if self.publish_message_history_to_weave and isinstance(
            self.message_history, genai.ChatSession
        ):
            history = []
            for message in self.message_history.history:
                for part in message.parts:
                    history.append({"role": message.role, "text": part.text})
            weave.publish(
                weave.MessagesPrompt(history),
                name="medqa_message_history_gemini",
            )

        if not isinstance(self.message_history, genai.ChatSession):
            self.message_history = model.start_chat()

        generation_config = (
            None
            if schema is None
            else genai.GenerationConfig(
                response_mime_type="application/json", response_schema=schema
            )
        )
        return get_chat_response(self.message_history, user_prompt, generation_config)

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

        if self.publish_system_prompt_to_weave:
            ref = weave.publish(
                weave.MessagesPrompt(system_messages),
                name="medqa_system_prompt_mistral",
            )
            system_messages = (
                weave.ref(
                    f"weave:///{ref.entity}/{ref.project}/object/{ref.name}:{ref._digest}"
                )
                .get()
                .format()
            )

        messages = []
        messages = (
            messages + [{"role": "system", "content": system_messages}]
            if len(system_messages) > 0
            else messages
        )
        messages = (
            messages + [{"role": "user", "content": user_messages}]
            if len(user_messages) > 0
            else messages
        )
        self.message_history.extend(messages)

        if self.publish_message_history_to_weave:
            weave.publish(
                weave.MessagesPrompt(messages),
                name="medqa_message_history_mistral",
            )

        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        client = instructor.from_mistral(client) if schema is not None else client

        if schema is None:
            raise NotImplementedError(
                "Mistral does not support structured output using a schema"
            )
        else:
            response = client.chat.complete(
                model=self.model_name, messages=self.message_history
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

        if self.publish_system_prompt_to_weave:
            ref = weave.publish(
                weave.MessagesPrompt(system_messages), name="medqa_system_prompt_openai"
            )
            system_messages = (
                weave.ref(
                    f"weave:///{ref.entity}/{ref.project}/object/{ref.name}:{ref._digest}"
                )
                .get()
                .format()
            )

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
        self.message_history.extend(messages)

        if self.publish_message_history_to_weave:
            weave.publish(
                weave.MessagesPrompt(messages),
                name="medqa_message_history_openai",
            )

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        if schema is None:
            completion = client.chat.completions.create(
                model=self.model_name, messages=self.message_history
            )
            return completion.choices[0].message.content

        completion = client.beta.chat.completions.parse(
            model=self.model_name,
            messages=self.message_history,
            response_format=schema,
        )
        response = completion.choices[0].message.parsed
        self.message_history.append({"role": "assistant", "content": response})
        return response

    @weave.op()
    def predict(
        self,
        user_prompt: Union[str, list[str]],
        system_prompt: Optional[Union[str, list[str]]] = None,
        schema: Optional[Any] = None,
    ) -> Union[str, Any]:
        """
        Predicts the response from a language model based on the provided prompts and schema.

        This function determines the client type and calls the appropriate SDK execution function
        to get the response from the language model. It supports multiple client types including
        GEMINI, MISTRAL, and OPENAI. Depending on the client type, it calls the corresponding
        execution function with the provided user and system prompts, and an optional schema.

        Args:
            user_prompt (Union[str, list[str]]): The user prompt(s) to be sent to the language model.
            system_prompt (Optional[Union[str, list[str]]]): The system prompt(s) to be sent to the language model.
            schema (Optional[Any]): The schema to be used for parsing the response, if applicable.

        Returns:
            Union[str, Any]: The response from the language model, which could be a string or any other type
            depending on the schema provided.

        Raises:
            ValueError: If the client type is invalid.
        """
        if self.client_type == ClientType.GEMINI:
            return self.execute_gemini_sdk(user_prompt, system_prompt, schema)
        elif self.client_type == ClientType.MISTRAL:
            return self.execute_mistral_sdk(user_prompt, system_prompt, schema)
        elif self.client_type == ClientType.OPENAI:
            return self.execute_openai_sdk(user_prompt, system_prompt, schema)
        else:
            raise ValueError(f"Invalid client type: {self.client_type}")
