import requests
from langchain.embeddings.base import Embeddings

class LLMStudioEmbeddings(Embeddings):
    def __init__(self, api_url: str):
        self.api_url = api_url

    def embed_query(self, text: str):
        response = requests.post(f"{self.api_url}/v1/embeddings", json={"input": text})
        response_data = response.json()
        return response_data['data'][0]['embedding']

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
