import os
import ollama
from openai import OpenAI

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
        try:
            client = OpenAI(api_key=self.apikey)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": global_context},
                    {"role": "user", "content": customer_prompt}
                ]
            )
            out = response.choices[0].message.content
            return out
        except ValueError as e:
            print(e)
            return None



    

    
    