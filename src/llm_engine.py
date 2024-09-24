from typing import List, Dict, Any
import json
import requests
import aiohttp
import sseclient


FIREWORKS_API_ENDPOINT = "https://api.fireworks.ai/inference/v1"
TOGETHER_API_ENDPOINT = "https://api.together.xyz/v1"
OPENAI_API_ENDPOINT = "https://api.openai.com/v1"
LOCAL_API_ENDPOINT = "http://localhost:8000/v1"
HEADERS = {
  "accept": "application/json",
  "content-type": "application/json",
}


class LLMEngine:
  def __init__(self, api_endpoint: str = None, provider: str = None, api_key: str = None):
    assert api_endpoint != None or provider != None
    if api_endpoint is None:
        if provider == 'openai':
            self.api_endpoint = OPENAI_API_ENDPOINT
        elif provider == 'fireworks':
            self.api_endpoint = FIREWORKS_API_ENDPOINT
        elif provider == 'together':
            self.api_endpoint = TOGETHER_API_ENDPOINT
        elif provider == 'localhost':
            self.api_endpoint = LOCAL_API_ENDPOINT
    else:
        self.api_endpoint = api_endpoint
    self.headers = HEADERS.copy()
    self.headers.update({"Authorization": f"Bearer {api_key}"})

  def generate(
      self,
      messages: List[Dict[str, str]],
      model: str,
      temperature: float = None,
      top_p: float = None,
      max_new_tokens: int = None,
      stop_sequences: List[str] = None,
      **kwargs
  ):
    payload = {
      "model": model,
      "messages": messages,
      "max_tokens": max_new_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "stream": False,
      "stop": stop_sequences,
    }
    response = requests.post(f'{self.api_endpoint}/chat/completions', json=payload, headers=self.headers)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

  async def agenerate(
    self,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = None,
    top_p: float = None,
    max_new_tokens: int = None,
    stop_sequences: List[str] = None,
    **kwargs
  ):
    payload = {
      "model": model,
      "messages": messages,
      "max_tokens": max_new_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "stream": False,
      "stop": stop_sequences,
    }
    
    async with aiohttp.ClientSession() as session:
      async with session.post(f'{self.api_endpoint}/chat/completions', json=payload, headers=self.headers) as response:
        response.raise_for_status()
        response_json = await response.json()
        return response_json['choices'][0]['message']['content']

  def completion(
      self,
      prompt: str,
      model: str,
      temperature: float = None,
      top_p: float = None,
      max_new_tokens: int = None,
      stop_sequences: List[str] = None,
      **kwargs
  ):
    payload = {
      "model": model,
      "prompt": prompt,
      "max_tokens": max_new_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "stream": False,
      "stop": stop_sequences
    }
    response = requests.post(f'{self.api_endpoint}/completions', json=payload, headers=self.headers)
    return response.json()['choices'][0]['text']
