import argparse
from transformers import pipeline
from openai import OpenAI
import re, random, time, json, replicate, os, httpx, threading
import pandas as pd
import openpyxl


def query_model(model_str, prompt, system_prompt, base_url=None, api_key=None, tries=5, timeout=30, max_prompt_len=2**14, clip_prompt=False):
    global _SHARED_CLIENT, _CLIENT_API_KEY, _CLIENT_BASE_URL

    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ['NO_PROXY'] = 'zjuqx.cn'
    
    # 如果没传入 base_url 或 api_key，使用环境变量
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.zjuqx.cn/v1")
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,  # 使用传入的地址
        http_client=httpx.Client(verify=True, timeout=httpx.Timeout(180.0, connect=60.0), follow_redirects=True))

    # 请求循环
    if clip_prompt: prompt = prompt[:max_prompt_len]

    for _ in range(tries):
        try:
            request_params = {
                "model": model_str,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.01,
            }

            if "Huatuo" in model_str: #对huatuo的限制
                request_params["max_tokens"] = 800 

            response = client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content or ""
            total_tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
            
            # 合并空格但保留换行
            answer = re.sub(r"[ \t]+", " ", content).strip()
            return answer, total_tokens

        except Exception as e:
            print(f"[DEBUG] API Error: {type(e).__name__} - {e}", flush=True)
            time.sleep(timeout)
    
    raise TimeoutError("Max retries reached")

# 参数名和数据集的对应关系
class Scenario:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.patient_id = scenario_dict["id"] #第几个病历
        self.diagnosis = scenario_dict["Diagnosis"] #病的正确答案
        self.patient_info  = scenario_dict["Medical_Record"] #病人信息
        self.examiner_info  = scenario_dict["Clinical_Findings"] #检查信息
        self.others = scenario_dict["Disease_Info"] #其他信息
    
    def patient_id(self):
        return self.patient_id
    def diagnosis_information(self) -> dict:
        return self.diagnosis    
    def patient_information(self) -> dict:
        return self.patient_info
    def examiner_information(self) -> dict:
        return self.examiner_info
    def other_information(self) -> dict:
        return self.others

# 调用数据集
class ScenarioLoader:
    def __init__(self, dataset_path: str) -> None:
        # 定义数据集简称与路径的对应关系
        DATA_MAP = {
            'pmc': "../data/pmc100.json",
            'clinic': "../data/clinic100.json",
            'mtmed': "../data/mtmed100.json"
        }
        if dataset_path in DATA_MAP:
            final_path = DATA_MAP[dataset_path]
        else:
            raise ValueError(f"数据集不可用")

        print(f"正在加载数据集: {final_path}")

        with open(final_path, "r", encoding="utf-8") as f: 
            self.scenario_strs = json.load(f)
            
        self.scenarios = [Scenario(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

