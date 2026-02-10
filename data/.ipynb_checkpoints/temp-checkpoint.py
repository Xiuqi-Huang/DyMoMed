#这个代码是将数据统一成我们想要的格式的

import os
import json
import time
import httpx
from openai import OpenAI, APIConnectionError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 1. 稳定性配置：彻底禁用系统代理，防止干扰自定义 base_url
os.environ['NO_PROXY'] = 'zjuqx.cn'
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# 2. 初始化客户端 (跳过 SSL 验证并增加超时时间)
http_client = httpx.Client(verify=False, timeout=120.0)
client = OpenAI(
    api_key="sk-sgQIJ9TOzf9ikrSYhNHlUTuIP9vVzBvljVwTU7SwS9YYX3wP", 
    base_url="https://api.zjuqx.cn/v1",
    http_client=http_client 
)

# 3. 自动重试逻辑：防止网络抖动导致的 Connection Error
@retry(
    stop=stop_after_attempt(4), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
)
def call_openai_with_retry(prompt):
    return client.chat.completions.create(
        model="gpt-4.1",  #注意这里是使用的大模型！
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

def generate_prompt(index, case_json):

    # original_diagnosis = case_json.get("OSCE_Examination", {}).get("Correct_Diagnosis", "Unknown")#针对agentclinic
    # case_title = case_json.get("title", "Unknown Title") #针对PMC700
    case_title = case_json.get("diagnosis", "Unknown Title") #针对MTMed700


    return f"""
    You are a medical assistant. Restructure the medical case into a standardized JSON object.
    
    Requirements:
    1. **Output Format**: A valid JSON object with keys: [id, Medical_Record, Clinical_Findings, Diagnosis, Department].
    2. **Language**: English.
    3. **Diagnosis Rule**:
       - Primary Source: Analyze the content: "{case_title}".
       - Extraction: Extract the specific disease name.
    4. **Department Selection**: Choose one and only one department from the following list based on the case:
       [Internal Medicine, Surgery, Obstetrics and Gynecology, Pediatrics, Psychiatry, Emergency Medicine, General Practice].
    5. **Atomic & Objective Style**:
       - Use short phrases (e.g., "Sharp chest pain" instead of "The patient feels a sharp pain").
       - Remove all personal pronouns (I, He, She, My, The patient).
    6. **Missing Data Rule**: If a specific item is not mentioned in the input, do NOT invent information. Leave the field as an empty string "".
    
    Structure Definitions:
    - **Medical_Record**: 
      - Demographics: Age, sex, occupation.
      - Chief_Complaint: Primary reason for visit + duration.
      - HPI: Detailed symptoms, onset, and progression.
      - Background: Comprehensive synthesis of PMH, surgical history, family/social history, medications, allergies, and other contextual data (immunizations, travel, etc.).
    - **Clinical_Findings**: 
      - Physical_Examination: Vital signs and systematic physical findings.
      - Investigations: Laboratory tests, imaging (CT/MRI/X-ray), and biopsy.
    - **Diagnosis**: Use the words from the content: "{case_title}".
    - **Department**: Categorize into one of the 7 specified medical departments.

    -----
    Example Output:
    {{
      "id": {index},
      "Medical_Record": {{
        "Demographics": "30-year-old female; software developer",
        "Chief_Complaint": "Acute RLQ abdominal pain; 12-hour duration",
        "HPI": "Sudden onset sharp pain; progressively worsening; nausea; no vomiting",
        "Background": "No significant PMH; non-smoker; no drug allergies; family history of hypertension"
      }},
      "Clinical_Findings": {{
        "Physical_Examination": "Temp 37.2°C; RLQ tenderness; Positive Rovsing's sign; No abdominal distension",
        "Investigations": "WBC 12,000/μL (elevated); Ultrasound shows enlarged appendix (7mm) with wall thickening"
      }},
      "Diagnosis": "Acute Appendicitis",
      "Department": "Surgery"
    }}
    -----

    Target Input:
    {json.dumps(case_json)}

    Target Output (with id = {index}):
    """

def process_cases(input_path, output_path, limit=None):
    processed_data = []
    
    # 读取并过滤空行
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]

    target_lines = all_lines[:limit] if limit is not None else all_lines
    total = len(target_lines)
    print(f"Processing {total} records...")

    for idx, line in enumerate(target_lines):
        try:
            raw_data = json.loads(line)
            prompt = generate_prompt(idx, raw_data)

            # 调用带重试逻辑的模型
            response = call_openai_with_retry(prompt)

            content = response.choices[0].message.content
            if content:
                # 清理可能是 Markdown 格式的返回
                clean_content = content.replace("```json", "").replace("```", "").strip()
                case_structured = json.loads(clean_content)
                processed_data.append(case_structured)
                print(f"[{idx + 1}/{total}] Case {idx} processed.")
            
            # 适当间隔防止频率限制
            time.sleep(0.5)

        except Exception as e:
            print(f"[{idx + 1}/{total}] Case {idx} failed: {e}")

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(processed_data, f_out, indent=4, ensure_ascii=False)
    print(f"\n Done! Saved {len(processed_data)} cases to {output_path}")

# --- 运行示例 ---
if __name__ == "__main__":
    # process_cases("data/agentclinic.jsonl", "data/pro_agentclinic.json")
    # process_cases("data/PMC700.jsonl", "data/pro_PMC.json")
    process_cases("MTMed700.jsonl", "pro_MTMed.json")



