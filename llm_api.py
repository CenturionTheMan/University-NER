from abc import ABC, abstractmethod
from typing import Optional
import requests
import time
from google import genai
from google.genai import types
import time

class LLMApi(ABC):
    def __init__(self, key: str, model: str):
        if not key:
            raise ValueError("API key must be provided")
        
        if not model:
            raise ValueError("Model must be provided")

        self.key = key
        self.model = model

    @abstractmethod
    def send_message(
        self, message: str, is_reasoning: bool = False, temperature: float = 0.7, max_tokens: Optional[int] = 2048, attempts_amt: int = 5) -> str:
        pass



class GeminiApi(LLMApi):
    def __init__(self, key: str, model: str = "gemini-2.5-flash-lite"):
        super().__init__(key, model)
        self.client = genai.Client(api_key=key)

    def send_message(self, message: str, is_reasoning: bool = False, temperature: float = 0.7, max_tokens: int | None = 2048, attempts_amt: int = 5) -> str:

        config = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        if is_reasoning:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=1024)

        for attempt in range(attempts_amt):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=message,
                    config=types.GenerateContentConfig(**config),
                )

                if response.text:
                    return response.text.strip()

                print("Empty Gemini response, retrying...")

            except Exception as e:
                wait = 2 ** attempt
                print(f"Gemini API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        raise Exception("Max retries exceeded")
    


class OpenRouterApi(LLMApi):
    def __init__(self, key: str, model: str):
        super().__init__(key, model)

    def send_message(self, message: str, is_reasoning: bool = False, temperature: float = 0.7, max_tokens: int|None = 2048, attempts_amt: int = 5) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
            "temperature": temperature,
            "reasoning": {"enabled": is_reasoning},
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "<YOUR_SITE_URL>",
            "X-Title": "<YOUR_SITE_NAME>",
        }

        for attempt in range(attempts_amt):
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if content is None:
                    print(f"Empty response, retrying... ({data})")
                    continue
                return content.strip()

            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            raise Exception(f"API error: {response.status_code} - {response.text}")

        raise Exception("Max retries exceeded")