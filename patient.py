from utils import query_model

class PatientAgent:
    def __init__(self, scenario, backend_str, base_url=None, api_key=None) -> None:
        # self.disease = "" #之后如果用不到就删了
        self.symptoms = "" #病人病历
        self.examiner_info = "" #检查信息
        self.agent_hist = "" #对话历史
        self.backend = backend_str #病人使用的llm
        self.scenario = scenario #一条数据
        self.token_cost = 0 # 初始化token计数
        self.reset()
        self.pipe = None
        self.base_url = base_url
        self.api_key = api_key


    def inference_patient(self, question) -> str: # 输入是医生的提问，除此之外还会补充对话历史，医生的回应，角色设定prompt
        user_prompt = f"""
            Dialogue history:
            {self.agent_hist}
            Doctor's utterances:{question}
            Please respond to the doctor now:
            """
        answer, tokens = query_model(self.backend, user_prompt, self.system_prompt(), self.base_url, self.api_key)
        self.token_cost += tokens
        self.agent_hist += "Doctor: " + question + "\n" + "Patient: " + answer + "\n"
        return answer
    
    def inference_second_patient(self, question, retry_msg) -> str: # 输入是监测器返回的内容
        
        marker = "Doctor: " + question
        last_turn_start = self.agent_hist.rfind(marker)
        if last_turn_start != -1:
            self.agent_hist = self.agent_hist[:last_turn_start] #覆盖之前错的问答对

        user_prompt = f"""
            Dialogue history: 
            {self.agent_hist}
            Feedback on your last utterances: 
            {retry_msg}
            Current Doctor's utterances: {question}
            Please provide your corrected response to the doctor now:
            """
        answer, tokens = query_model(self.backend, user_prompt, self.system_prompt())
        self.token_cost += tokens
        self.agent_hist += "Doctor: " + question + "\n" + "Patient: " + answer + "\n"
        return answer
    
    def system_prompt(self) -> str: # 角色设定的prompt
        base = f"""
            You are a patient in a clinic undergoing a medical consultation and assessment by a doctor. 

            Instructions:
            1. Personality: You must speak according to the Personality in the medical record.
            2. Accuracy: Use ONLY provided medical record and clinical findings. If unsure, say "I don't know." Do not make up information.
            3. Privacy: Never name the Diagnosis.
            4. Constraints: Speak as 'I' in 1–3 sentences. ONLY output patient dialogue. NO meta-talk.

            Your medical record: 
            {self.symptoms} 

            Your clinical findings: 
            {self.examiner_info} 
            """
        return base
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()
        self.examiner_info =self.scenario.examiner_information()

class PatientMonitorAgent: 
    def __init__(self, scenario, backend_str,base_url=None, api_key=None) -> None:
        self.disease = "" #正确的病
        self.symptoms = "" #病人病历
        self.examiner_info = "" #检查信息
        self.backend = backend_str #使用的llm
        self.scenario = scenario #一条数据
        self.token_cost = 0 # 初始化token计数
        self.reset()
        self.pipe = None
        self.base_url = base_url
        self.api_key = api_key
    
    def inference_patient(self, question) -> str: 
        user_prompt = f"""
            Patient's utterances: 
            {question}
            Please evaluate the content above:
            """
        answer,tokens = query_model(self.backend, user_prompt, self.system_prompt(),self.base_url, self.api_key)
        self.token_cost += tokens
        return answer
    
    def system_prompt(self) -> str: # 角色设定的prompt
        base = f"""
            Role: Patient Response Monitor. 
            Task: Validate if the patient's utterance complies with the evaluation criteria and the provided patient data. 

            Evaluation criteria:
            1. Constraints: Must use 'I' and be 1-3 sentences. ONLY output patient dialogue. NO meta-talk.
            2. Privacy: NEVER name the Diagnosis.
            3. Accuracy: Use ONLY provided medical record and clinical findings. If unsure, say "I don't know." Do not make up information.
            4. Personality Alignment: MUST strictly embody the assigned emotion and speaking style in the 'Personality' field in the medical record. Character-driven minor deviations from other rules (except Privacy) are acceptable and prioritized.

            Output logic:
            If PASS: Reply with 'OK' and NOTHING ELSE.
            If FAIL: ONLY address the violated criteria and provide a short suggestion. NO detailed analysis.

            Patient data:
            Medical Record: {self.symptoms}
            Clinical Findings: {self.examiner_info}
            Diagnosis: {self.disease}
            """
        return base

    def reset(self) -> None:
        self.symptoms = self.scenario.patient_information()
        self.examiner_info = self.scenario.examiner_information()
        self.disease = self.scenario.diagnosis_information()