# development_notebook.ipynb
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configure llm-d client
class LLMDClient:
    def __init__(self, endpoint, model_name):
        self.endpoint = endpoint
        self.model_name = model_name
        self.session = requests.Session()
    
    def chat_completion(self, messages, **kwargs):
        """Send chat completion request to llm-d"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": kwargs.get("stream", False)
        }
        
        response = self.session.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    
    def embeddings(self, input_text, model=None):
        """Generate embeddings for input text"""
        payload = {
            "model": model or self.model_name,
            "input": input_text
        }
        
        response = self.session.post(
            f"{self.endpoint}/v1/embeddings",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return response.json()

# Initialize client
client = LLMDClient(
    endpoint="http://localhost:8080",  # Port-forwarded service
    model_name="llama3-8b-dev"
)

# Test connection
test_response = client.chat_completion([
    {"role": "user", "content": "Hello! Please respond with a simple greeting."}
])
print(json.dumps(test_response, indent=2))