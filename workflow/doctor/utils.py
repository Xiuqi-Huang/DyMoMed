import argparse
from transformers import pipeline
from openai import OpenAI
import re, random, time, json, replicate, os, httpx, threading
import pandas as pd


#暂时用不到
def load_huggingface_model(model_name):
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe

#暂时用不到
def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response

# 全局变量
_SHARED_CLIENT = None
_CLIENT_API_KEY = None   
_CLIENT_BASE_URL = None  
_CLIENT_LOCK = threading.Lock() 

def query_model(model_str, prompt, system_prompt, base_url=None, api_key=None, tries=30, timeout=20.0, max_prompt_len=2**14, clip_prompt=False):
    global _SHARED_CLIENT, _CLIENT_API_KEY, _CLIENT_BASE_URL

    # 1. 确定最终使用的配置 (优先参数，后环境变量)
    target_base_url = base_url if base_url else os.environ.get("OPENAI_BASE_URL", "")
    target_api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")

    # 2. 线程安全的 Client 初始化/更新
    if (_SHARED_CLIENT is None or 
        target_api_key != _CLIENT_API_KEY or 
        target_base_url != _CLIENT_BASE_URL):
        
        with _CLIENT_LOCK:
            if (_SHARED_CLIENT is None or 
                target_api_key != _CLIENT_API_KEY or 
                target_base_url != _CLIENT_BASE_URL):
                
                os.environ.update({"http_proxy": "", "https_proxy": "", "NO_PROXY": "zjuqx.cn"})
                
                _SHARED_CLIENT = OpenAI(
                    api_key=target_api_key,
                    base_url=target_base_url,
                    http_client=httpx.Client(verify=True, timeout=httpx.Timeout(300.0, connect=60.0), follow_redirects=True)
                )
                
                _CLIENT_API_KEY = target_api_key
                _CLIENT_BASE_URL = target_base_url

    # 3. 模型 ID 处理
    if "HF_" in model_str:
        raise Exception("HF models TODO")

    # 4. 请求循环
    if clip_prompt: prompt = prompt[:max_prompt_len]

    for _ in range(tries):
        try:
            # 1. 准备通用的请求参数
            request_params = {
                "model": model_str, # 保持使用你的 model_str
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.01,
            }

            if "Huatuo" in model_str: #对huatuo的限制
                request_params["max_tokens"] = 800 

            # 3. 使用解包参数的方式发起请求
            response = _SHARED_CLIENT.chat.completions.create(**request_params)
            
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
        self.department = scenario_dict["Department"] #病所属的部门
    
    def patient_id(self):
        return self.patient_id
    def diagnosis_information(self) -> dict:
        return self.diagnosis    
    def patient_information(self) -> dict:
        return self.patient_info
    def examiner_information(self) -> dict:
        return self.examiner_info
    def department_information(self) -> dict:
        return self.department

# 调用数据集
class ScenarioLoader:
    def __init__(self, dataset_path: str) -> None:
        # 定义数据集简称与路径的对应关系
        DATA_MAP = {
            'pmc': "../data/emo_pmc50.json",
            'clinic': "../data/emo_clinic214.json",
            'mtmed': "../data/emo_mtmed50.json"
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
