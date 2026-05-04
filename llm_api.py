from abc import ABC, abstractmethod
from typing import Dict, List
import requests
import json


class LLMApi(ABC):
    @abstractmethod
    def send_message(self, message:str) -> str:
        pass
    
class OpenRouterApi(LLMApi):
    def __init__(self, key: str, model:str):
        if not key:
            raise ValueError("API key must be provided")
        
        if not model:
            raise ValueError("Model must be provided")
        
        self.key = key
        self.model = model
        
    def send_message(self, message: str) -> str:
        data_str = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": message
                }],
            "reasoning": {"enabled": True}
            })

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
            data=data_str
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        response_data = response.json()
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            raise Exception("No choices found in response")
        return response_data['choices'][0]['message']['content'].strip()