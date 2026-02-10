# 医生的agent
from utils import query_model
import re

class DoctorAgent:
    def __init__(self, scenario, backend_str, max_infs=10, base_url=None, api_key=None) -> None:

        self.infs = 0 # 当前医生进行了多少次提问
        self.MAX_INFS = max_infs # 医生可以发起提问的次数
        self.agent_hist = "" # 对话历史
        self.backend = backend_str # 医生使用的llm
        self.scenario = scenario # 一条数据
        self.token_cost = 0 # 初始化token计数
        self.reset()
        self.pipe = None
        self.base_url = base_url
        self.api_key = api_key

    def inference_doctor(self, question) -> str: # uu：医生每轮更新的话，内含prompt+对话历史+病人的回答+医生的这一轮要说的话

        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        user_prompt = f"""
            Dialogue history:
            {self.agent_hist}
            Patient's utterances: {question}
            Please respond to the patient now:
            """
        answer, tokens = query_model(self.backend, user_prompt, self.system_prompt(),self.base_url, self.api_key)
        
        self.token_cost += tokens
        
        parsed_output = {} 
        patterns = {
             "thought": r"(?:Thought|Thinking|Think)[:\n]\s*(.*?)(?=\s*(?:ACTION|Response|Diagnosis|Treatment|##\s*Final\s*Response):|$)",
            "action": r"ACTION:\s*(.*?)(?=\s*(?:Thought|Thinking|Think|Response|Diagnosis|Treatment|##):|$)",
            "response": r"(?:Response:|##\s*Final\s*Response)\s*(.*?)(?=\s*(?:Thought|Thinking|Think|ACTION|Diagnosis|Treatment):|$)",
            "diagnosis": r"Diagnosis:\s*(.*?)(?=\s*(?:Thought|Thinking|Think|ACTION|Response|Treatment|##):|$)",
             "treatment": r"Treatment:\s*(.*?)(?=\s*(?:Thought|Thinking|Think|ACTION|Response|Diagnosis|##):|$)"}

        for key, pattern in patterns.items():
            match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
            if match:
                parsed_output[key] = match.group(1).strip()
        
        # 提取response给病人
        true_answer = parsed_output.get("response", answer).strip()
        
        self.agent_hist += "Patient: " + question + "\n" + "Doctor: " + true_answer + "\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str: # uu：医生的prompt，要说诊断和建议
        base = f"""
            Output format:
            Thought: [Clinical reasoning and decision rationale]
            ACTION: [continue | diagnose]
            Response: [Your professional utterances to the patient]
            Diagnosis: [The specific disease name if diagnosing; otherwise, omit this field]
            Treatment: [The treatment plan if diagnosing; otherwise, omit this field]
            """
        return base

    def reset(self) -> None:
        self.agent_hist = ""