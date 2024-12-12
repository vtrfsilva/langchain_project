import requests
from langchain.llms.base import LLM

class CustomLLM(LLM):
    api_url: str
    message_history: list = []

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.message_history = []

    def _call(self, prompt: str, stop=None) -> str:
        self.message_history.append({"role": "user", "content": prompt})
        response = requests.post(self.api_url, json={"messages": self.message_history})
        response_data = response.json()
        assistant_message = response_data['choices'][0]['message']['content']
        self.message_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    @property
    def _llm_type(self) -> str:
        return "custom"
