import os

from typing import Dict, List
from abc import ABC, abstractmethod
from openai import OpenAI

import google.generativeai as genai
import anthropic






class ModelConfig:
    def __init__(self, model_type: str, model_name: str):
        self.model_type = model_type # 'gpt', 'gemini', 'claude', 'ds', 'qwen'
        self.model_name = model_name




# 基类
class BaseAgent(ABC):    
    def __init__(self, name: str, model_config: ModelConfig):

        self.name = name
        self.model_config = model_config
        self.model_type = model_config.model_type
        self.model_name = model_config.model_name

        self._init_client()


    def _init_client(self):
        
        if self.model_type == 'gpt':
            self.client = OpenAI(api_key=os.environ.get('GPT_API_KEY'))

        elif self.model_type == 'gemini':
            genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
            self.model = genai.GenerativeModel(self.model_name)
            
        elif self.model_type == 'claude':
            self.client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY'))
            
        elif self.model_type == 'ds':
            self.client = OpenAI(api_key=os.environ.get('DS_API_KEY'))
            
        elif self.model_type == 'qwen':
            self.client = OpenAI(api_key=os.environ.get('QWEN_API_KEY'))

        else:
            raise ValueError(f"错误模型类别: {self.model_type}")
        

    # 生成一次回复
    def call_llm(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})
        
        try:
            return self._call_api(messages, temperature)
        except Exception as e:
            print(f"智能体 {self.name} 调用call_llm函数时发生异常: {e}")
            return f"异常: {e}"
        

    def _call_api(self, messages: List[Dict], temperature: float) -> str:
        
        # gpt
        if self.model_type == 'gpt':
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
            
        # gemini
        elif self.model_type == 'gemini':
            # 合并为纯str
            combined_prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    combined_prompt += f"[System Prompt: {msg['content']}]\n\n"
                elif msg['role'] == 'user':
                    combined_prompt += msg['content']
            
            response = self.model.generate_content(
                combined_prompt,
                generation_config=genai.GenerationConfig(temperature=temperature)
            )
            
            return response.text
                
        # claude
        elif self.model_type == 'claude':
            system_message = ""
            claude_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                elif msg['role'] == 'user':
                    claude_messages.append({"role": "user", "content": msg['content']})
            
            response = self.client.messages.create(
                model=self.model_name,
                messages=claude_messages,
                system=system_message if system_message else None,
                temperature=temperature,
            )
            
            return response.content[0].text
            
        # ds
        elif self.model_type == 'ds':
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
            
        # qwen
        elif self.model_type == 'qwen':
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        
        else:
            return f"错误模型类别: {self.model_type}"


    # 给出回复前的处理
    @abstractmethod
    def respond(self, message: dict, **kwargs) -> str:
        pass