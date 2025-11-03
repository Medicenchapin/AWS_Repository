import requests
import os
import ollama
from openai import OpenAI
from together import Together


class Model:
    
    def __init__(self, model, url, apikey):
        self.model = model
        self.url = url
        self.apikey = apikey

    def call_llm(self, global_context: str, customer_prompt: str) -> str:
        full_prompt = [
            {'role': 'system', 'content': global_context},
            {'role': 'user', 'content': customer_prompt}
        ]
        out: str = ''
        for chunk in ollama.chat(self.model, messages=full_prompt, stream=True):
            content = chunk["message"]["content"]
            # print(content, end="", flush=True)
            out += content        
        return out
        
        
    def call_openAI(self, global_context: str, customer_prompt:str):
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": global_context},
                {"role": "user", "content": customer_prompt}
            ]
        )

        out = response.choices[0].message.content
        return out
    
class TogetherAIModel: 
    def __init__(self, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 1024, stop=None):
        # Acepta TOGETHER_API_KEY o TOGETHERAI_API_KEY (como en tu .env)
        api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Falta TOGETHER_API_KEY/TOGETHERAI_API_KEY en el entorno.")
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop

    def call_llm(self, global_context: str, customer_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": global_context},
                {"role": "user",   "content": customer_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )
        return resp.choices[0].message.content


    

    
    