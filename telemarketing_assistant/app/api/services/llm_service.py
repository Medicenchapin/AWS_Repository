import os
import requests
from app.api.core import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

def run_llm(prompt: str) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant for telemarketing strategy."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 600
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]