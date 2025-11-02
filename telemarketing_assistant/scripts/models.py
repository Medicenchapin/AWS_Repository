import requests
import os
import ollama
from openai import OpenAI

class Model:
    
    def __init__(self, model, url, apikey):
        self.model = model
        self.url = url
        self.apikey = apikey
        
        pass


    def call_llm(self, global_context: str, customer_prompt: str) -> str:
        full_prompt = [
            {'role': 'system', 'content': global_context},
            {'role': 'user', 'content': customer_prompt}
        ]
        out: str = ''
        for chunk in ollama.chat(OLLAMA_MODEL, messages=full_prompt, stream=True):
            print(chunk['message']['content'], end='', flush=True)
        # return out.strip()
        
        
    def call_openAI(self, global_context: str, customer_prompt:str):
        
        # Configura el endpoint de Ollama como si fuera OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="none")

        response = client.chat.completions.create(
            model="cas/nous-hermes-2-mistral-7b-dpo",
            messages=[
                {"role": "system", "content": "Eres un analista de telecomunicaciones."},
                {"role": "user", "content": "Genera un resumen SHAP para este cliente: ..."}
            ]
        )

        print(response.choices[0].message.content)



    

    
    